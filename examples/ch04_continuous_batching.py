#!/usr/bin/env python3
"""Dependency-free teaching model for vLLM V1 token scheduling.

The model keeps the contracts explained in Chapter 4: dynamic arrivals,
RUNNING-before-WAITING scheduling, token/input/sequence budgets, chunked
prefill, FCFS or priority ordering, KV-pressure preemption, and the separate
schedule/update phases. It is intentionally smaller than the real scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


class SchedulingPolicy(str, Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"


@dataclass
class Request:
    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    arrival_step: int = 0
    priority: int = 0
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0
    output_tokens: list[int] = field(default_factory=list)
    kv_tokens: int = 0
    num_preemptions: int = 0
    first_scheduled_step: int | None = None
    finished_step: int | None = None

    def __post_init__(self) -> None:
        if self.prompt_tokens < 1:
            raise ValueError("prompt_tokens must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

    @property
    def num_tokens(self) -> int:
        """Current target: prompt plus output tokens already produced."""

        return self.prompt_tokens + len(self.output_tokens)

    @property
    def deficit(self) -> int:
        """Tokens the model still needs to compute to catch up to the target."""

        return max(self.num_tokens - self.num_computed_tokens, 0)

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.prompt_tokens


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    num_tokens: int
    phase: str
    was_new: bool
    was_resumed: bool


@dataclass(frozen=True)
class SchedulerOutput:
    step: int
    scheduled: tuple[ScheduledRequest, ...]
    preempted: tuple[str, ...]
    token_budget_used: int
    input_budget_used: int

    @property
    def num_scheduled_tokens(self) -> dict[str, int]:
        return {item.request_id: item.num_tokens for item in self.scheduled}


class TeachingScheduler:
    """Small state machine that exposes the decisions made in each step."""

    def __init__(
        self,
        *,
        max_num_scheduled_tokens: int,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int = 4,
        kv_capacity_tokens: int = 4096,
        enable_chunked_prefill: bool = True,
        long_prefill_token_threshold: int = 0,
        policy: SchedulingPolicy = SchedulingPolicy.FCFS,
        draft_slots_per_request: int = 0,
        reserve_full_prompt: bool = True,
    ) -> None:
        if max_num_scheduled_tokens < 1:
            raise ValueError("max_num_scheduled_tokens must be positive")
        if max_num_batched_tokens is None:
            max_num_batched_tokens = max_num_scheduled_tokens
        if max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be positive")
        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")
        if kv_capacity_tokens < 1:
            raise ValueError("kv_capacity_tokens must be positive")
        if draft_slots_per_request < 0:
            raise ValueError("draft_slots_per_request cannot be negative")

        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.kv_capacity_tokens = kv_capacity_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.policy = policy
        self.draft_slots_per_request = draft_slots_per_request
        self.reserve_full_prompt = reserve_full_prompt

        self.step_id = 0
        self.requests: dict[str, Request] = {}
        self.pending_arrivals: list[Request] = []
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.history: list[SchedulerOutput] = []

    def add_request(self, request: Request) -> None:
        if request.request_id in self.requests:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        self.requests[request.request_id] = request
        self.pending_arrivals.append(request)
        self.pending_arrivals.sort(key=lambda item: (item.arrival_step, item.request_id))

    def _admit_arrivals(self) -> None:
        arrived = [r for r in self.pending_arrivals if r.arrival_step <= self.step_id]
        self.pending_arrivals = [
            r for r in self.pending_arrivals if r.arrival_step > self.step_id
        ]
        self.waiting.extend(arrived)

    def _waiting_order(self) -> list[Request]:
        if self.policy == SchedulingPolicy.PRIORITY:
            return sorted(
                self.waiting,
                key=lambda r: (r.priority, r.arrival_step, r.request_id),
            )
        return list(self.waiting)

    @property
    def used_kv_tokens(self) -> int:
        return sum(request.kv_tokens for request in self.running)

    def _victim(self) -> Request:
        if self.policy == SchedulingPolicy.PRIORITY:
            return max(
                self.running,
                key=lambda r: (r.priority, r.arrival_step, r.request_id),
            )
        return self.running[-1]

    def _preempt(self, request: Request) -> None:
        self.running.remove(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.kv_tokens = 0
        request.num_preemptions += 1
        if self.policy == SchedulingPolicy.FCFS:
            self.waiting.insert(0, request)
        else:
            self.waiting.append(request)

    def _fit_running_allocation(
        self, request: Request, num_new_tokens: int
    ) -> tuple[bool, list[str]]:
        preempted: list[str] = []
        needed = max(request.num_computed_tokens + num_new_tokens - request.kv_tokens, 0)
        while self.used_kv_tokens + needed > self.kv_capacity_tokens:
            if not self.running:
                return False, preempted
            victim = self._victim()
            preempted.append(victim.request_id)
            self._preempt(victim)
            if victim is request:
                return False, preempted
            needed = max(
                request.num_computed_tokens + num_new_tokens - request.kv_tokens, 0
            )
        request.kv_tokens += needed
        return True, preempted

    def _waiting_allocation_sizes(
        self, request: Request, num_new_tokens: int
    ) -> tuple[int, int]:
        allocation = max(
            request.num_computed_tokens + num_new_tokens - request.kv_tokens, 0
        )
        fit_requirement = allocation
        if self.reserve_full_prompt:
            fit_requirement = max(
                fit_requirement, request.prompt_tokens - request.kv_tokens
            )
        return allocation, fit_requirement

    def _schedule_running(
        self,
        token_budget: int,
        input_budget: int,
    ) -> tuple[list[ScheduledRequest], list[str], int, int]:
        scheduled: list[ScheduledRequest] = []
        preempted: list[str] = []
        index = 0
        while index < len(self.running) and token_budget > 0:
            if input_budget <= self.draft_slots_per_request:
                break
            request = self.running[index]
            deficit = request.deficit
            if deficit == 0:
                index += 1
                continue
            num_new_tokens = deficit
            if (
                self.long_prefill_token_threshold > 0
                and num_new_tokens > self.long_prefill_token_threshold
            ):
                num_new_tokens = self.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens,
                token_budget,
                input_budget - self.draft_slots_per_request,
            )
            if num_new_tokens <= 0:
                index += 1
                continue

            fitted, victims = self._fit_running_allocation(request, num_new_tokens)
            preempted.extend(victims)
            for victim_id in victims:
                prior = next(
                    (item for item in scheduled if item.request_id == victim_id), None
                )
                if prior is not None:
                    scheduled.remove(prior)
                    token_budget += prior.num_tokens
                    input_budget += (
                        prior.num_tokens + self.draft_slots_per_request
                    )
            if not fitted:
                break
            scheduled.append(
                ScheduledRequest(
                    request.request_id,
                    num_new_tokens,
                    "prefill" if request.is_prefill else "decode",
                    was_new=False,
                    was_resumed=False,
                )
            )
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + self.draft_slots_per_request
            index += 1
        return scheduled, preempted, token_budget, input_budget

    def _schedule_waiting(
        self,
        token_budget: int,
        input_budget: int,
    ) -> tuple[list[ScheduledRequest], int, int]:
        scheduled: list[ScheduledRequest] = []
        while self.waiting and token_budget > 0:
            if input_budget <= self.draft_slots_per_request:
                break
            if len(self.running) >= self.max_num_seqs:
                break

            request = self._waiting_order()[0]
            request_budget = min(
                token_budget, input_budget - self.draft_slots_per_request
            )
            num_new_tokens = request.deficit
            if (
                self.long_prefill_token_threshold > 0
                and num_new_tokens > self.long_prefill_token_threshold
            ):
                num_new_tokens = self.long_prefill_token_threshold
            if not self.enable_chunked_prefill and num_new_tokens > request_budget:
                break
            num_new_tokens = min(num_new_tokens, request_budget)
            if num_new_tokens <= 0:
                break

            allocation, fit_requirement = self._waiting_allocation_sizes(
                request, num_new_tokens
            )
            if self.used_kv_tokens + fit_requirement > self.kv_capacity_tokens:
                break

            self.waiting.remove(request)
            was_resumed = request.status == RequestStatus.PREEMPTED
            request.status = RequestStatus.RUNNING
            request.kv_tokens += allocation
            request.first_scheduled_step = (
                self.step_id
                if request.first_scheduled_step is None
                else request.first_scheduled_step
            )
            self.running.append(request)
            scheduled.append(
                ScheduledRequest(
                    request.request_id,
                    num_new_tokens,
                    "prefill" if request.is_prefill else "decode",
                    was_new=not was_resumed,
                    was_resumed=was_resumed,
                )
            )
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + self.draft_slots_per_request
        return scheduled, token_budget, input_budget

    def schedule(self) -> SchedulerOutput:
        self._admit_arrivals()
        token_budget = self.max_num_scheduled_tokens
        input_budget = self.max_num_batched_tokens

        running, preempted, token_budget, input_budget = self._schedule_running(
            token_budget, input_budget
        )
        waiting: list[ScheduledRequest] = []
        if not preempted:
            waiting, token_budget, input_budget = self._schedule_waiting(
                token_budget, input_budget
            )

        scheduled = running + waiting
        for item in scheduled:
            request = self.requests[item.request_id]
            request.num_computed_tokens += item.num_tokens

        output = SchedulerOutput(
            step=self.step_id,
            scheduled=tuple(scheduled),
            preempted=tuple(preempted),
            token_budget_used=self.max_num_scheduled_tokens - token_budget,
            input_budget_used=self.max_num_batched_tokens - input_budget,
        )
        self.history.append(output)
        return output

    def update_from_output(self, output: SchedulerOutput) -> None:
        """Append one sampled token only when this step caught up to the target."""

        for item in output.scheduled:
            request = self.requests[item.request_id]
            if request.status != RequestStatus.RUNNING:
                continue
            if request.num_computed_tokens < request.num_tokens:
                continue
            if len(request.output_tokens) < request.max_new_tokens:
                token_id = 1000 + len(request.output_tokens)
                request.output_tokens.append(token_id)
            if len(request.output_tokens) == request.max_new_tokens:
                request.status = RequestStatus.FINISHED
                request.finished_step = self.step_id
                request.kv_tokens = 0
                self.running.remove(request)
                self.finished.append(request)
        self.step_id += 1

    def step(self) -> SchedulerOutput:
        output = self.schedule()
        self.update_from_output(output)
        return output

    def has_work(self) -> bool:
        return bool(self.pending_arrivals or self.waiting or self.running)

    def run(self, max_steps: int = 100) -> list[SchedulerOutput]:
        while self.has_work() and self.step_id < max_steps:
            self.step()
        if self.has_work():
            raise RuntimeError("scheduler did not drain within max_steps")
        return self.history

    def format_step(self, output: SchedulerOutput) -> str:
        work = ", ".join(
            f"{item.request_id}:{item.phase[:1]}{item.num_tokens}"
            for item in output.scheduled
        ) or "idle"
        victims = ",".join(output.preempted) or "-"
        waiting = ",".join(r.request_id for r in self._waiting_order()) or "-"
        running = ",".join(r.request_id for r in self.running) or "-"
        return (
            f"step={output.step:02d} work=[{work}] preempted=[{victims}] "
            f"running=[{running}] waiting=[{waiting}] "
            f"kv={self.used_kv_tokens}/{self.kv_capacity_tokens}"
        )


def demo() -> None:
    scheduler = TeachingScheduler(
        max_num_scheduled_tokens=8,
        max_num_seqs=3,
        kv_capacity_tokens=64,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=6,
        reserve_full_prompt=False,
    )
    scheduler.add_request(Request("A", prompt_tokens=12, max_new_tokens=3))
    scheduler.add_request(
        Request("B", prompt_tokens=3, max_new_tokens=2, arrival_step=1)
    )
    scheduler.add_request(
        Request("C", prompt_tokens=2, max_new_tokens=2, arrival_step=2)
    )

    while scheduler.has_work():
        output = scheduler.step()
        print(scheduler.format_step(output))


if __name__ == "__main__":
    demo()
