#!/usr/bin/env python3
"""Dependency-free teaching model for vLLM's model-execution stack.

The implementation deliberately models contracts, not GPU performance.  It
keeps Executor broadcast/reply behavior, Worker ownership, persistent request
rows, ragged input preparation, the execute/sample state hand-off, pipeline
intermediate values, and a small but correctly ordered sampling pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from random import Random
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


@dataclass(frozen=True)
class NewRequest:
    req_id: str
    prompt_token_ids: tuple[int, ...]
    sampling_params: SamplingParams = SamplingParams()


@dataclass(frozen=True)
class SchedulerStep:
    """Small equivalent of the execution-relevant SchedulerOutput fields."""

    scheduled_tokens: dict[str, int]
    new_requests: tuple[NewRequest, ...] = ()
    finished_req_ids: frozenset[str] = frozenset()
    preempted_req_ids: frozenset[str] = frozenset()

    @property
    def total_num_scheduled_tokens(self) -> int:
        return sum(self.scheduled_tokens.values())


@dataclass
class RequestState:
    req_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    num_computed_tokens: int
    row: int
    sampling_params: SamplingParams

    @property
    def all_token_ids(self) -> list[int]:
        return [*self.prompt_token_ids, *self.output_token_ids]


@dataclass(frozen=True)
class PreparedBatch:
    req_ids: list[str]
    row_indices: list[int]
    input_ids: list[int]
    positions: list[int]
    query_start_loc: list[int]


@dataclass(frozen=True)
class ForwardState:
    batch: PreparedBatch
    logits: list[list[float]] | None
    intermediate: list[float] | None


@dataclass(frozen=True)
class IntermediateTensors:
    values: list[float]


@dataclass(frozen=True)
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]


def select_executor_backend(
    backend: str, *, use_ray_v2: bool = False
) -> str:
    """Mirror the important branches in Executor.get_class."""

    if backend == "uni":
        return "UniProcExecutor"
    if backend == "mp":
        return "MultiprocExecutor"
    if backend == "ray":
        return "RayExecutorV2" if use_ray_v2 else "RayDistributedExecutor"
    if backend == "external_launcher":
        return "ExecutorWithExternalLauncher"
    if "." in backend:
        return backend
    raise ValueError(f"unknown executor backend: {backend}")


def resolve_model_class(
    architectures: Sequence[str], registry: dict[str, str]
) -> tuple[str, str]:
    """Return the first registered architecture and its lazy class target."""

    if not architectures:
        raise ValueError("no model architectures are specified")
    for architecture in architectures:
        if architecture in registry:
            return architecture, registry[architecture]
    raise ValueError(f"unsupported architectures: {list(architectures)}")


def choose_loader(load_format: str) -> str:
    loaders = {
        "auto": "DefaultModelLoader",
        "hf": "DefaultModelLoader",
        "safetensors": "DefaultModelLoader",
        "dummy": "DummyModelLoader",
        "sharded_state": "ShardedStateLoader",
        "tensorizer": "TensorizerLoader",
    }
    try:
        return loaders[load_format]
    except KeyError as exc:
        raise ValueError(f"unsupported load format: {load_format}") from exc


def build_query_start_loc(lengths: Iterable[int]) -> list[int]:
    starts = [0]
    for length in lengths:
        if length < 0:
            raise ValueError("query length must be non-negative")
        starts.append(starts[-1] + length)
    return starts


def prepare_batch(
    states: dict[str, RequestState], scheduled_tokens: dict[str, int]
) -> PreparedBatch:
    """Flatten request-local token slices into one ragged execution batch."""

    req_ids: list[str] = []
    row_indices: list[int] = []
    input_ids: list[int] = []
    positions: list[int] = []
    lengths: list[int] = []
    for req_id, count in scheduled_tokens.items():
        if count <= 0:
            raise ValueError("scheduled token count must be positive")
        state = states[req_id]
        start = state.num_computed_tokens
        stop = start + count
        tokens = state.all_token_ids[start:stop]
        if len(tokens) != count:
            raise ValueError(
                f"request {req_id} has {len(tokens)} available tokens, needs {count}"
            )
        req_ids.append(req_id)
        row_indices.append(state.row)
        input_ids.extend(tokens)
        positions.extend(range(start, stop))
        lengths.append(count)
    return PreparedBatch(
        req_ids=req_ids,
        row_indices=row_indices,
        input_ids=input_ids,
        positions=positions,
        query_start_loc=build_query_start_loc(lengths),
    )


def _softmax(logits: Sequence[float]) -> list[float]:
    maximum = max(logits)
    numerators = [exp(value - maximum) for value in logits]
    total = sum(numerators)
    return [value / total for value in numerators]


def process_logits(
    logits: Sequence[float],
    history: Sequence[int],
    params: SamplingParams,
) -> list[float]:
    """Apply penalties, temperature, top-k and top-p in teaching order."""

    processed = [float(value) for value in logits]
    counts = {token_id: history.count(token_id) for token_id in set(history)}
    for token_id, count in counts.items():
        if 0 <= token_id < len(processed):
            processed[token_id] -= params.presence_penalty
            processed[token_id] -= params.frequency_penalty * count

    if params.temperature == 0:
        return processed
    processed = [value / params.temperature for value in processed]

    if params.top_k and params.top_k < len(processed):
        keep = set(
            sorted(range(len(processed)), key=processed.__getitem__, reverse=True)[
                : params.top_k
            ]
        )
        processed = [value if index in keep else float("-inf")
                     for index, value in enumerate(processed)]

    if params.top_p < 1:
        probabilities = _softmax(processed)
        ranked = sorted(
            range(len(processed)), key=probabilities.__getitem__, reverse=True
        )
        keep: set[int] = set()
        cumulative = 0.0
        for index in ranked:
            keep.add(index)
            cumulative += probabilities[index]
            if cumulative >= params.top_p:
                break
        processed = [value if index in keep else float("-inf")
                     for index, value in enumerate(processed)]
    return processed


def sample_token(
    logits: Sequence[float],
    history: Sequence[int],
    params: SamplingParams,
    *,
    sample_position: int,
) -> int:
    processed = process_logits(logits, history, params)
    if params.temperature == 0:
        return max(range(len(processed)), key=processed.__getitem__)

    probabilities = _softmax(processed)
    random_value = Random(params.seed + sample_position).random()
    cumulative = 0.0
    for token_id, probability in enumerate(probabilities):
        cumulative += probability
        if random_value <= cumulative:
            return token_id
    return len(probabilities) - 1


class TinyModel:
    """Deterministic logical model used by every replica."""

    def __init__(self, vocab_size: int = 8):
        self.vocab_size = vocab_size

    def forward(self, batch: PreparedBatch) -> list[list[float]]:
        logits = []
        for req_index in range(len(batch.req_ids)):
            end = batch.query_start_loc[req_index + 1]
            last_token = batch.input_ids[end - 1]
            position = batch.positions[end - 1]
            center = (last_token * 3 + position + 1) % self.vocab_size
            logits.append(
                [-abs(token_id - center) for token_id in range(self.vocab_size)]
            )
        return logits


class PersistentModelRunner:
    """Small MRV2-like runner with permanent request rows and ephemeral state."""

    def __init__(self, max_num_reqs: int = 8, *, is_last_pp_rank: bool = True):
        self.model = TinyModel()
        self.max_num_reqs = max_num_reqs
        self.is_last_pp_rank = is_last_pp_rank
        self.requests: dict[str, RequestState] = {}
        self.free_rows = list(range(max_num_reqs))
        self.execute_model_state: ForwardState | None = None

    def _remove(self, req_id: str) -> None:
        state = self.requests.pop(req_id, None)
        if state is not None:
            self.free_rows.append(state.row)
            self.free_rows.sort()

    def _update_requests(self, step: SchedulerStep) -> None:
        for req_id in step.finished_req_ids | step.preempted_req_ids:
            self._remove(req_id)
        for request in step.new_requests:
            if request.req_id in self.requests:
                raise ValueError(f"duplicate request: {request.req_id}")
            if not self.free_rows:
                raise RuntimeError("no free persistent request rows")
            self.requests[request.req_id] = RequestState(
                req_id=request.req_id,
                prompt_token_ids=list(request.prompt_token_ids),
                output_token_ids=[],
                num_computed_tokens=0,
                row=self.free_rows.pop(0),
                sampling_params=request.sampling_params,
            )

    def execute_model(
        self, step: SchedulerStep, intermediate: IntermediateTensors | None = None
    ) -> IntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError("sample_tokens must consume the previous execute state")
        self._update_requests(step)
        if step.total_num_scheduled_tokens == 0:
            return None

        batch = prepare_batch(self.requests, step.scheduled_tokens)
        if self.is_last_pp_rank:
            logits = self.model.forward(batch)
            self.execute_model_state = ForwardState(batch, logits, None)
            return None

        values = [float(token + position)
                  for token, position in zip(batch.input_ids, batch.positions)]
        if intermediate is not None:
            values = [left + right for left, right in zip(values, intermediate.values)]
        self.execute_model_state = ForwardState(batch, None, values)
        return IntermediateTensors(values)

    def sample_tokens(self) -> ModelRunnerOutput:
        state = self.execute_model_state
        self.execute_model_state = None
        if state is None:
            return ModelRunnerOutput([], {}, [])
        if not self.is_last_pp_rank:
            return ModelRunnerOutput([], {}, [])
        assert state.logits is not None

        sampled: list[list[int]] = []
        for req_id, logits in zip(state.batch.req_ids, state.logits):
            request = self.requests[req_id]
            history = request.all_token_ids
            token_id = sample_token(
                logits,
                history,
                request.sampling_params,
                sample_position=len(history),
            )
            request.num_computed_tokens += (
                state.batch.query_start_loc[len(sampled) + 1]
                - state.batch.query_start_loc[len(sampled)]
            )
            request.output_token_ids.append(token_id)
            sampled.append([token_id])

        req_ids = list(state.batch.req_ids)
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
        )


class WorkerReplica:
    def __init__(self, rank: int, *, is_last_pp_rank: bool = True):
        self.rank = rank
        self.model_runner = PersistentModelRunner(is_last_pp_rank=is_last_pp_rank)
        self.received_steps: list[SchedulerStep] = []

    def execute_model(self, step: SchedulerStep) -> IntermediateTensors | None:
        self.received_steps.append(step)
        return self.model_runner.execute_model(step)

    def sample_tokens(self) -> ModelRunnerOutput:
        return self.model_runner.sample_tokens()


class MultiprocExecutorModel:
    """Broadcast every call but read a unique rank's reply."""

    def __init__(self, world_size: int, output_rank: int = 0):
        if not 0 <= output_rank < world_size:
            raise ValueError("output_rank is outside world_size")
        self.workers = [WorkerReplica(rank) for rank in range(world_size)]
        self.output_rank = output_rank

    def execute_model(self, step: SchedulerStep) -> None:
        for worker in self.workers:
            worker.execute_model(step)

    def sample_tokens(self) -> ModelRunnerOutput:
        replies = [worker.sample_tokens() for worker in self.workers]
        return replies[self.output_rank]


def demo() -> None:
    executor = MultiprocExecutorModel(world_size=2, output_rank=0)
    step = SchedulerStep(
        new_requests=(NewRequest("A", (1, 2, 3)), NewRequest("B", (4,))),
        scheduled_tokens={"A": 3, "B": 1},
    )
    executor.execute_model(step)
    output = executor.sample_tokens()
    print(output)


if __name__ == "__main__":
    demo()
