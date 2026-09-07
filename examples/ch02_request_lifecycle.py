#!/usr/bin/env python3
"""A dependency-free model of the vLLM V1 request lifecycle.

This is a teaching model, not a performance simulator.  It preserves the
important ownership boundaries used in Chapter 2:

raw input -> EngineInput -> EngineCoreRequest -> SchedulerOutput
          -> ModelRunnerOutput -> EngineCoreOutput -> RequestOutput
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass(frozen=True)
class EngineInput:
    prompt: str
    prompt_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class EngineCoreRequest:
    request_id: str
    external_request_id: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int


@dataclass(frozen=True)
class SchedulerOutput:
    request_ids: tuple[str, ...]
    num_scheduled_tokens: dict[str, int]


@dataclass(frozen=True)
class ModelRunnerOutput:
    sampled_token_ids: dict[str, int]


@dataclass(frozen=True)
class EngineCoreOutput:
    request_id: str
    new_token_ids: tuple[int, ...]
    finish_reason: str | None = None


@dataclass(frozen=True)
class RequestOutput:
    request_id: str
    text: str
    token_ids: tuple[int, ...]
    finished: bool
    finish_reason: str | None


class TinyTokenizer:
    """Whitespace tokenizer with a fixed vocabulary for deterministic output."""

    def __init__(self) -> None:
        pieces = [
            "<bos>",
            "请",
            "介绍",
            "vLLM",
            "它",
            "让",
            "推理",
            "更",
            "高效",
            "。",
        ]
        self.token_to_id = {piece: index for index, piece in enumerate(pieces)}
        self.id_to_token = {index: piece for piece, index in self.token_to_id.items()}

    def encode(self, text: str) -> tuple[int, ...]:
        pieces = text.split()
        if not pieces:
            raise ValueError("prompt cannot be empty")
        try:
            return (self.token_to_id["<bos>"],) + tuple(
                self.token_to_id[piece] for piece in pieces
            )
        except KeyError as error:
            raise ValueError(f"unknown teaching token: {error.args[0]}") from error

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self.id_to_token[token_id] for token_id in token_ids)


class Renderer:
    def __init__(self, tokenizer: TinyTokenizer) -> None:
        self.tokenizer = tokenizer

    def render(self, prompt: str) -> EngineInput:
        return EngineInput(prompt=prompt, prompt_token_ids=self.tokenizer.encode(prompt))


class InputProcessor:
    def __init__(self, max_model_len: int = 16) -> None:
        self.max_model_len = max_model_len
        self._sequence = 0

    def process(
        self, external_request_id: str, engine_input: EngineInput, max_tokens: int
    ) -> EngineCoreRequest:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if len(engine_input.prompt_token_ids) + max_tokens > self.max_model_len:
            raise ValueError("prompt plus output exceeds max_model_len")
        self._sequence += 1
        internal_id = f"{external_request_id}-internal-{self._sequence:02d}"
        return EngineCoreRequest(
            request_id=internal_id,
            external_request_id=external_request_id,
            prompt_token_ids=engine_input.prompt_token_ids,
            max_tokens=max_tokens,
        )


@dataclass
class CoreRequestState:
    request: EngineCoreRequest
    planned_token_ids: tuple[int, ...]
    status: RequestStatus = RequestStatus.WAITING
    cursor: int = 0


class EngineCore:
    """One-token-per-request scheduler plus deterministic model runner."""

    def __init__(self, response_plan: tuple[int, ...]) -> None:
        self.response_plan = response_plan
        self.requests: dict[str, CoreRequestState] = {}
        self.abort_outputs: deque[EngineCoreOutput] = deque()

    def add_request(self, request: EngineCoreRequest) -> None:
        self.requests[request.request_id] = CoreRequestState(
            request=request,
            planned_token_ids=self.response_plan[: request.max_tokens],
        )

    def abort(self, request_id: str) -> None:
        state = self.requests.get(request_id)
        if state is not None and state.status not in {
            RequestStatus.FINISHED,
            RequestStatus.ABORTED,
        }:
            state.status = RequestStatus.ABORTED
            self.abort_outputs.append(
                EngineCoreOutput(
                    request_id=request_id,
                    new_token_ids=(),
                    finish_reason="abort",
                )
            )

    def schedule(self) -> SchedulerOutput:
        runnable = []
        for request_id, state in self.requests.items():
            if state.status in {RequestStatus.WAITING, RequestStatus.RUNNING}:
                state.status = RequestStatus.RUNNING
                runnable.append(request_id)
        return SchedulerOutput(
            request_ids=tuple(runnable),
            num_scheduled_tokens={request_id: 1 for request_id in runnable},
        )

    def execute_model(self, scheduled: SchedulerOutput) -> ModelRunnerOutput:
        sampled = {}
        for request_id in scheduled.request_ids:
            state = self.requests[request_id]
            sampled[request_id] = state.planned_token_ids[state.cursor]
        return ModelRunnerOutput(sampled_token_ids=sampled)

    def update_from_output(
        self, model_output: ModelRunnerOutput
    ) -> list[EngineCoreOutput]:
        outputs = []
        for request_id, token_id in model_output.sampled_token_ids.items():
            state = self.requests[request_id]
            if state.status == RequestStatus.ABORTED:
                outputs.append(
                    EngineCoreOutput(
                        request_id=request_id,
                        new_token_ids=(),
                        finish_reason="abort",
                    )
                )
                continue
            state.cursor += 1
            finished = state.cursor == len(state.planned_token_ids)
            if finished:
                state.status = RequestStatus.FINISHED
            outputs.append(
                EngineCoreOutput(
                    request_id=request_id,
                    new_token_ids=(token_id,),
                    finish_reason="length" if finished else None,
                )
            )
        return outputs

    def step(self) -> list[EngineCoreOutput]:
        pending_aborts = list(self.abort_outputs)
        self.abort_outputs.clear()
        scheduled = self.schedule()
        if not scheduled.request_ids:
            return pending_aborts
        model_output = self.execute_model(scheduled)
        return pending_aborts + self.update_from_output(model_output)

    def has_unfinished_requests(self) -> bool:
        return any(
            state.status in {RequestStatus.WAITING, RequestStatus.RUNNING}
            for state in self.requests.values()
        )


@dataclass
class FrontendState:
    external_request_id: str
    token_ids: list[int] = field(default_factory=list)
    queue: deque[RequestOutput] = field(default_factory=deque)


class OutputProcessor:
    def __init__(self, tokenizer: TinyTokenizer) -> None:
        self.tokenizer = tokenizer
        self.states: dict[str, FrontendState] = {}

    def add_request(self, request: EngineCoreRequest) -> None:
        self.states[request.request_id] = FrontendState(request.external_request_id)

    def process(
        self, outputs: list[EngineCoreOutput], *, stream: bool
    ) -> list[RequestOutput]:
        completed = []
        for output in outputs:
            state = self.states.get(output.request_id)
            if state is None:
                continue
            state.token_ids.extend(output.new_token_ids)
            request_output = RequestOutput(
                request_id=state.external_request_id,
                text=self.tokenizer.decode(output.new_token_ids if stream else tuple(state.token_ids)),
                token_ids=output.new_token_ids if stream else tuple(state.token_ids),
                finished=output.finish_reason is not None,
                finish_reason=output.finish_reason,
            )
            if stream:
                state.queue.append(request_output)
            if request_output.finished:
                completed.append(request_output)
                del self.states[output.request_id]
        return completed


class LifecycleDemo:
    def __init__(self) -> None:
        self.tokenizer = TinyTokenizer()
        self.renderer = Renderer(self.tokenizer)
        self.input_processor = InputProcessor()
        response = tuple(
            self.tokenizer.token_to_id[token]
            for token in ("它", "让", "推理", "更", "高效", "。")
        )
        self.core = EngineCore(response)
        self.output_processor = OutputProcessor(self.tokenizer)

    def submit(self, request_id: str, prompt: str, max_tokens: int) -> str:
        engine_input = self.renderer.render(prompt)
        request = self.input_processor.process(request_id, engine_input, max_tokens)
        # Register frontend state before the request crosses the core boundary.
        self.output_processor.add_request(request)
        self.core.add_request(request)
        return request.request_id

    def run_offline(self) -> list[RequestOutput]:
        results = []
        while self.core.has_unfinished_requests():
            results.extend(self.output_processor.process(self.core.step(), stream=False))
        return sorted(results, key=lambda output: output.request_id)

    def tick_online(self) -> list[RequestOutput]:
        queues = [state.queue for state in self.output_processor.states.values()]
        self.output_processor.process(self.core.step(), stream=True)
        chunks = []
        for queue in queues:
            while queue:
                chunks.append(queue.popleft())
        return chunks


def main() -> None:
    offline = LifecycleDemo()
    offline.submit("req-A", "请 介绍 vLLM", max_tokens=3)
    offline.submit("req-B", "介绍 vLLM", max_tokens=6)
    print("offline final outputs:")
    for output in offline.run_offline():
        print(f"  {output.request_id}: {output.text} ({output.finish_reason})")

    online = LifecycleDemo()
    internal_id = online.submit("chatcmpl-42", "请 介绍 vLLM", max_tokens=6)
    print("online stream:")
    for step in range(1, 4):
        chunks = online.tick_online()
        print(f"  step {step}: {[chunk.text for chunk in chunks]}")
    online.core.abort(internal_id)
    aborted = online.tick_online()
    print(f"  abort: {[(chunk.text, chunk.finish_reason) for chunk in aborted]}")


if __name__ == "__main__":
    main()
