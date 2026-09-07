#!/usr/bin/env python3
"""Dependency-free teaching model for the vLLM V1 EngineCore loop.

The model intentionally keeps only the control-flow contracts discussed in
Chapter 3: client selection, schedule/execute/update, the concurrent batch
queue, abort ordering, pause modes, shutdown, and executor failure.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class ClientKind(str, Enum):
    INPROC = "InprocClient"
    SYNC_MP = "SyncMPClient"
    ASYNC_MP = "AsyncMPClient"


def select_client(*, multiprocess: bool, asyncio_mode: bool) -> ClientKind:
    """Mirror EngineCoreClient.make_client's three supported branches."""

    if asyncio_mode and not multiprocess:
        raise NotImplementedError("async EngineCore requires multiprocessing")
    if multiprocess and asyncio_mode:
        return ClientKind.ASYNC_MP
    if multiprocess:
        return ClientKind.SYNC_MP
    return ClientKind.INPROC


class PauseState(str, Enum):
    UNPAUSED = "unpaused"
    PAUSED_NEW = "paused_new"
    PAUSED_ALL = "paused_all"


class ShutdownState(str, Enum):
    RUNNING = "running"
    REQUESTED = "requested"
    SHUTTING_DOWN = "shutting_down"


class MessageType(str, Enum):
    ADD = "add"
    ABORT = "abort"
    UTILITY = "utility"
    EXECUTOR_FAILED = "executor_failed"
    WAKEUP = "wakeup"


@dataclass
class RequestState:
    request_id: str
    max_tokens: int
    launched_tokens: int = 0
    settled_tokens: int = 0
    finished: bool = False
    aborted: bool = False
    admitted_before_pause: bool = True


@dataclass(frozen=True)
class SchedulerOutput:
    batch_id: int
    request_id: str
    token_position: int
    total_num_scheduled_tokens: int = 1


@dataclass(frozen=True)
class ModelRunnerOutput:
    request_id: str
    token_position: int
    token_id: int


@dataclass(frozen=True)
class EngineCoreOutput:
    request_id: str
    token_ids: tuple[int, ...]
    finish_reason: str | None = None


class TinyScheduler:
    """Round-robin, one-token batches with explicit launched/settled state."""

    def __init__(self) -> None:
        self.requests: dict[str, RequestState] = {}
        self.order: deque[str] = deque()
        self.pause_state = PauseState.UNPAUSED
        self._next_batch_id = 1

    def add_request(self, request_id: str, max_tokens: int) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if request_id in self.requests:
            raise ValueError(f"duplicate request_id: {request_id}")
        self.requests[request_id] = RequestState(
            request_id=request_id,
            max_tokens=max_tokens,
            admitted_before_pause=self.pause_state == PauseState.UNPAUSED,
        )
        self.order.append(request_id)

    def _eligible(self, state: RequestState) -> bool:
        if state.finished or state.aborted:
            return False
        if state.launched_tokens >= state.max_tokens:
            return False
        if self.pause_state == PauseState.PAUSED_ALL:
            return False
        if self.pause_state == PauseState.PAUSED_NEW:
            return state.admitted_before_pause
        return True

    def can_schedule(self) -> bool:
        return any(self._eligible(state) for state in self.requests.values())

    def has_unfinished_requests(self) -> bool:
        return any(not state.finished and not state.aborted for state in self.requests.values())

    def schedule(self) -> SchedulerOutput | None:
        for _ in range(len(self.order)):
            request_id = self.order[0]
            self.order.rotate(-1)
            state = self.requests[request_id]
            if not self._eligible(state):
                continue
            position = state.launched_tokens
            state.launched_tokens += 1
            output = SchedulerOutput(self._next_batch_id, request_id, position)
            self._next_batch_id += 1
            return output
        return None

    def update_from_output(
        self, scheduled: SchedulerOutput, model: ModelRunnerOutput
    ) -> dict[int, tuple[EngineCoreOutput, ...]]:
        state = self.requests[scheduled.request_id]
        if state.aborted:
            return {}
        if model.request_id != scheduled.request_id:
            raise ValueError("model output belongs to another request")
        state.settled_tokens += 1
        finished = state.settled_tokens == state.max_tokens
        state.finished = finished
        output = EngineCoreOutput(
            request_id=state.request_id,
            token_ids=(model.token_id,),
            finish_reason="length" if finished else None,
        )
        return {0: (output,)}

    def abort(self, request_ids: list[str]) -> tuple[EngineCoreOutput, ...]:
        outputs = []
        for request_id in request_ids:
            state = self.requests.get(request_id)
            if state is None or state.finished or state.aborted:
                continue
            state.aborted = True
            outputs.append(EngineCoreOutput(request_id, (), "abort"))
        return tuple(outputs)

    def set_pause(self, state: PauseState) -> None:
        self.pause_state = state
        if state == PauseState.UNPAUSED:
            for request in self.requests.values():
                request.admitted_before_pause = True


class TinyExecutor:
    """Returns completed Futures, preserving the executor's async-shaped API."""

    def __init__(self) -> None:
        self.fail_next = False
        self.launched_batches: list[int] = []

    def execute_model(
        self, scheduled: SchedulerOutput, *, non_block: bool
    ) -> Future[ModelRunnerOutput]:
        if not non_block:
            raise AssertionError("the teaching path expects non_block=True")
        self.launched_batches.append(scheduled.batch_id)
        future: Future[ModelRunnerOutput] = Future()
        if self.fail_next:
            self.fail_next = False
            future.set_exception(RuntimeError("executor failed"))
        else:
            future.set_result(
                ModelRunnerOutput(
                    request_id=scheduled.request_id,
                    token_position=scheduled.token_position,
                    token_id=100 + scheduled.token_position,
                )
            )
        return future


BatchQueueItem = tuple[
    Future[ModelRunnerOutput], SchedulerOutput, Future[ModelRunnerOutput]
]


class TeachingEngineCore:
    def __init__(self, *, max_concurrent_batches: int = 1) -> None:
        if max_concurrent_batches < 1:
            raise ValueError("max_concurrent_batches must be positive")
        self.scheduler = TinyScheduler()
        self.executor = TinyExecutor()
        self.aborts_queue: deque[list[str]] = deque()
        self.batch_queue_size = max_concurrent_batches
        self.batch_queue: deque[BatchQueueItem] | None = (
            deque(maxlen=max_concurrent_batches)
            if max_concurrent_batches > 1
            else None
        )

    def add_request(self, request_id: str, max_tokens: int) -> None:
        self.scheduler.add_request(request_id, max_tokens)

    def queue_abort(self, *request_ids: str) -> None:
        self.aborts_queue.append(list(request_ids))

    def _process_aborts_queue(self) -> tuple[EngineCoreOutput, ...]:
        request_ids: list[str] = []
        while self.aborts_queue:
            request_ids.extend(self.aborts_queue.popleft())
        return self.scheduler.abort(request_ids)

    def step(
        self, after_launch: Callable[["TeachingEngineCore"], None] | None = None
    ) -> tuple[dict[int, tuple[EngineCoreOutput, ...]], bool]:
        scheduled = self.scheduler.schedule()
        if scheduled is None:
            return {}, False
        future = self.executor.execute_model(scheduled, non_block=True)
        if after_launch is not None:
            after_launch(self)
        model_output = future.result()
        abort_outputs = self._process_aborts_queue()
        outputs = self.scheduler.update_from_output(scheduled, model_output)
        if abort_outputs:
            outputs[0] = abort_outputs + outputs.get(0, ())
        return outputs, scheduled.total_num_scheduled_tokens > 0

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, tuple[EngineCoreOutput, ...]] | None, bool]:
        queue = self.batch_queue
        if queue is None:
            raise RuntimeError("batch queue is disabled")
        if len(queue) >= self.batch_queue_size:
            raise AssertionError("queue must be drained before scheduling")

        model_executed = False
        if self.scheduler.can_schedule():
            scheduled = self.scheduler.schedule()
            assert scheduled is not None
            future = self.executor.execute_model(scheduled, non_block=True)
            queue.appendleft((future, scheduled, future))
            model_executed = scheduled.total_num_scheduled_tokens > 0
            if len(queue) < self.batch_queue_size and self.scheduler.can_schedule():
                return None, model_executed

        elif not queue:
            return None, False

        future, scheduled, execute_future = queue.pop()
        model_output = future.result()
        if model_output is None:
            execute_future.result()
            raise RuntimeError("unexpected empty model output")
        abort_outputs = self._process_aborts_queue()
        outputs = self.scheduler.update_from_output(scheduled, model_output)
        if abort_outputs:
            outputs[0] = abort_outputs + outputs.get(0, ())
        return outputs, model_executed

    def pause(self, mode: str) -> tuple[EngineCoreOutput, ...]:
        if mode == "abort":
            outputs = self.scheduler.abort(list(self.scheduler.requests))
            self.scheduler.set_pause(PauseState.PAUSED_NEW)
            return outputs
        if mode == "wait":
            self.scheduler.set_pause(PauseState.PAUSED_NEW)
            return ()
        if mode == "keep":
            self.scheduler.set_pause(PauseState.PAUSED_ALL)
            return ()
        raise ValueError(f"invalid pause mode: {mode}")

    def resume(self) -> None:
        self.scheduler.set_pause(PauseState.UNPAUSED)

    def has_work(self) -> bool:
        return self.scheduler.has_unfinished_requests() or bool(self.batch_queue)


class TeachingEngineCoreProc:
    """Queue-facing shell that mirrors EngineCoreProc's dispatch decisions."""

    def __init__(self, core: TeachingEngineCore, *, shutdown_timeout: int = 0) -> None:
        self.core = core
        self.shutdown_timeout = shutdown_timeout
        self.shutdown_state = ShutdownState.RUNNING
        self.input_queue: deque[tuple[MessageType, object]] = deque()
        self.output_queue: deque[EngineCoreOutput | str] = deque()

    def submit(self, message_type: MessageType, payload: object = None) -> None:
        self.input_queue.append((message_type, payload))

    def request_shutdown(self) -> None:
        self.shutdown_state = ShutdownState.REQUESTED
        self.submit(MessageType.WAKEUP)

    def _handle_message(self, message_type: MessageType, payload: object) -> None:
        if message_type == MessageType.WAKEUP:
            return
        if message_type == MessageType.ADD:
            request_id, max_tokens = payload  # type: ignore[misc]
            if self.shutdown_state != ShutdownState.RUNNING:
                self.output_queue.append(EngineCoreOutput(request_id, (), "abort"))
            else:
                self.core.add_request(request_id, max_tokens)
            return
        if message_type == MessageType.ABORT:
            request_ids = list(payload)  # type: ignore[arg-type]
            self.core.queue_abort(*request_ids)
            return
        if message_type == MessageType.UTILITY:
            self.output_queue.append(f"utility:{payload}")
            return
        if message_type == MessageType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        raise ValueError(f"unknown message type: {message_type}")

    def run_once(self) -> bool:
        while self.input_queue:
            self._handle_message(*self.input_queue.popleft())

        if self.shutdown_state == ShutdownState.REQUESTED:
            if self.shutdown_timeout == 0:
                self.output_queue.extend(self.core.pause("abort"))
            self.shutdown_state = ShutdownState.SHUTTING_DOWN

        outputs, _ = (
            self.core.step_with_batch_queue()
            if self.core.batch_queue is not None
            else self.core.step()
        )
        if outputs:
            self.output_queue.extend(outputs.get(0, ()))

        return not (
            self.shutdown_state == ShutdownState.SHUTTING_DOWN
            and not self.core.has_work()
        )


def demo() -> None:
    core = TeachingEngineCore(max_concurrent_batches=2)
    core.add_request("req-0", max_tokens=3)
    for tick in range(4):
        outputs, launched = core.step_with_batch_queue()
        printable = None if outputs is None else outputs.get(0, ())
        print(
            f"tick={tick} launched={launched} "
            f"queue={len(core.batch_queue or ())} outputs={printable}"
        )


if __name__ == "__main__":
    demo()
