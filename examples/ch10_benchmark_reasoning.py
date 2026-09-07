#!/usr/bin/env python3
"""Dependency-free helpers for reasoning about vLLM benchmark results.

The module mirrors the metric boundaries used by ``vllm bench serve`` and
adds small experiment-analysis utilities. It does not send HTTP requests or
measure GPU kernels; feed it timestamps collected by a real benchmark.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Mapping, Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Return a linearly interpolated percentile, matching NumPy's default."""

    if not values:
        return 0.0
    if not 0 <= q <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class RequestTrace:
    """Client-observed timestamps for one streaming generation request."""

    arrival_time: float
    send_time: float
    token_times: tuple[float, ...]
    prompt_tokens: int
    success: bool = True
    error: str = ""

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if self.send_time < self.arrival_time:
            raise ValueError("send_time cannot precede client arrival")
        if any(
            later < earlier
            for earlier, later in zip(self.token_times, self.token_times[1:])
        ):
            raise ValueError("token timestamps must be non-decreasing")
        if self.token_times and self.token_times[0] < self.send_time:
            raise ValueError("a token cannot arrive before the request is sent")
        if self.success and not self.token_times:
            raise ValueError("a successful streaming request needs a token")

    @property
    def output_tokens(self) -> int:
        return len(self.token_times) if self.success else 0

    @property
    def client_queue_time(self) -> float:
        return self.send_time - self.arrival_time

    @property
    def ttft(self) -> float:
        if not self.success or not self.token_times:
            return 0.0
        return self.token_times[0] - self.send_time

    @property
    def itls(self) -> tuple[float, ...]:
        if not self.success:
            return ()
        return tuple(
            later - earlier
            for earlier, later in zip(self.token_times, self.token_times[1:])
        )

    @property
    def e2el(self) -> float:
        if not self.success or not self.token_times:
            return 0.0
        return self.token_times[-1] - self.send_time

    @property
    def tpot(self) -> float:
        """Average post-first-token interval for this request."""

        if self.output_tokens <= 1:
            return 0.0
        return (self.e2el - self.ttft) / (self.output_tokens - 1)


@dataclass(frozen=True)
class ServiceLevelObjectives:
    ttft_s: float | None = None
    tpot_s: float | None = None
    e2el_s: float | None = None

    def __post_init__(self) -> None:
        for value in (self.ttft_s, self.tpot_s, self.e2el_s):
            if value is not None and value < 0:
                raise ValueError("SLO values must be non-negative")

    def accepts(self, trace: RequestTrace) -> bool:
        if not trace.success:
            return False
        checks = (
            self.ttft_s is None or trace.ttft <= self.ttft_s,
            self.tpot_s is None or trace.tpot <= self.tpot_s,
            self.e2el_s is None or trace.e2el <= self.e2el_s,
        )
        return all(checks)


@dataclass(frozen=True)
class BenchmarkSummary:
    duration_s: float
    completed: int
    failed: int
    total_input_tokens: int
    total_output_tokens: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_s: float
    p50_ttft_s: float
    p99_ttft_s: float
    mean_tpot_s: float
    p99_tpot_s: float
    mean_itl_s: float
    p99_itl_s: float
    mean_e2el_s: float
    p99_e2el_s: float
    mean_client_queue_s: float


def summarize(
    traces: Sequence[RequestTrace],
    duration_s: float,
    slos: ServiceLevelObjectives | None = None,
) -> BenchmarkSummary:
    """Aggregate client-side request traces using serve-benchmark semantics."""

    if duration_s <= 0:
        raise ValueError("benchmark duration must be positive")
    successful = [trace for trace in traces if trace.success]
    ttfts = [trace.ttft for trace in successful]
    tpots = [trace.tpot for trace in successful if trace.output_tokens > 1]
    itls = [value for trace in successful for value in trace.itls]
    e2els = [trace.e2el for trace in successful]
    client_queues = [trace.client_queue_time for trace in successful]
    total_input = sum(trace.prompt_tokens for trace in successful)
    total_output = sum(trace.output_tokens for trace in successful)
    good = sum(slos.accepts(trace) for trace in successful) if slos else 0

    def mean(values: Sequence[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    return BenchmarkSummary(
        duration_s=duration_s,
        completed=len(successful),
        failed=len(traces) - len(successful),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        request_throughput=len(successful) / duration_s,
        request_goodput=good / duration_s,
        output_throughput=total_output / duration_s,
        total_token_throughput=(total_input + total_output) / duration_s,
        mean_ttft_s=mean(ttfts),
        p50_ttft_s=percentile(ttfts, 50),
        p99_ttft_s=percentile(ttfts, 99),
        mean_tpot_s=mean(tpots),
        p99_tpot_s=percentile(tpots, 99),
        mean_itl_s=mean(itls),
        p99_itl_s=percentile(itls, 99),
        mean_e2el_s=mean(e2els),
        p99_e2el_s=percentile(e2els, 99),
        mean_client_queue_s=mean(client_queues),
    )


@dataclass(frozen=True)
class ExperimentRun:
    name: str
    throughput_samples: tuple[float, ...]
    p99_latency_samples: tuple[float, ...]
    error_rate_samples: tuple[float, ...] = field(default_factory=tuple)
    config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.throughput_samples or not self.p99_latency_samples:
            raise ValueError("an experiment run needs throughput and latency samples")
        if any(value < 0 for value in self.throughput_samples):
            raise ValueError("throughput cannot be negative")
        if any(value < 0 for value in self.p99_latency_samples):
            raise ValueError("latency cannot be negative")
        if any(not 0 <= value <= 1 for value in self.error_rate_samples):
            raise ValueError("error rates must be between 0 and 1")

    @property
    def mean_throughput(self) -> float:
        return statistics.fmean(self.throughput_samples)

    @property
    def mean_p99_latency(self) -> float:
        return statistics.fmean(self.p99_latency_samples)

    @property
    def throughput_cv(self) -> float:
        if len(self.throughput_samples) < 2 or self.mean_throughput == 0:
            return 0.0
        return statistics.stdev(self.throughput_samples) / self.mean_throughput

    @property
    def mean_error_rate(self) -> float:
        return (
            statistics.fmean(self.error_rate_samples)
            if self.error_rate_samples
            else 0.0
        )


@dataclass(frozen=True)
class Comparison:
    throughput_change_pct: float
    p99_latency_change_pct: float
    baseline_cv: float
    treatment_cv: float
    passes_correctness_gate: bool


def compare_runs(
    baseline: ExperimentRun,
    treatment: ExperimentRun,
    max_error_rate: float = 0.0,
) -> Comparison:
    if baseline.mean_throughput == 0 or baseline.mean_p99_latency == 0:
        raise ValueError("baseline metrics must be non-zero")
    if max_error_rate < 0:
        raise ValueError("max_error_rate must be non-negative")
    return Comparison(
        throughput_change_pct=(
            treatment.mean_throughput / baseline.mean_throughput - 1
        )
        * 100,
        p99_latency_change_pct=(
            treatment.mean_p99_latency / baseline.mean_p99_latency - 1
        )
        * 100,
        baseline_cv=baseline.throughput_cv,
        treatment_cv=treatment.throughput_cv,
        passes_correctness_gate=treatment.mean_error_rate <= max_error_rate,
    )


def pareto_frontier(runs: Sequence[ExperimentRun]) -> list[ExperimentRun]:
    """Keep runs not dominated on throughput (high) and p99 latency (low)."""

    frontier = []
    for candidate in runs:
        dominated = any(
            other is not candidate
            and other.mean_throughput >= candidate.mean_throughput
            and other.mean_p99_latency <= candidate.mean_p99_latency
            and (
                other.mean_throughput > candidate.mean_throughput
                or other.mean_p99_latency < candidate.mean_p99_latency
            )
            for other in runs
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda run: run.mean_p99_latency)


@dataclass(frozen=True)
class RuntimeSignals:
    waiting_requests: int
    kv_usage: float
    preemptions: int
    gpu_busy_fraction: float
    communication_fraction: float
    cpu_gap_fraction: float

    def __post_init__(self) -> None:
        if self.waiting_requests < 0 or self.preemptions < 0:
            raise ValueError("request counts must be non-negative")
        for value in (
            self.kv_usage,
            self.gpu_busy_fraction,
            self.communication_fraction,
            self.cpu_gap_fraction,
        ):
            if not 0 <= value <= 1:
                raise ValueError("fractions must be between 0 and 1")


def diagnose(signals: RuntimeSignals) -> tuple[str, ...]:
    """Return evidence-driven hypotheses, not automatic root-cause claims."""

    hypotheses: list[str] = []
    if signals.kv_usage >= 0.9 and signals.preemptions > 0:
        hypotheses.append("kv_capacity_pressure")
    if signals.waiting_requests > 0 and signals.gpu_busy_fraction >= 0.9:
        hypotheses.append("gpu_saturated")
    if signals.communication_fraction >= 0.3:
        hypotheses.append("communication_heavy")
    if signals.cpu_gap_fraction >= 0.2 and signals.gpu_busy_fraction < 0.8:
        hypotheses.append("cpu_or_control_plane_gap")
    if not hypotheses:
        hypotheses.append("insufficient_or_balanced_signals")
    return tuple(hypotheses)


def demo() -> None:
    traces = [
        RequestTrace(0.0, 0.0, (0.10, 0.14, 0.18), 12),
        RequestTrace(0.0, 0.02, (0.14, 0.19), 20),
    ]
    summary = summarize(
        traces,
        duration_s=0.20,
        slos=ServiceLevelObjectives(ttft_s=0.15, tpot_s=0.06),
    )
    print("completed:", summary.completed)
    print("request throughput:", round(summary.request_throughput, 2), "req/s")
    print("output throughput:", round(summary.output_throughput, 2), "tok/s")
    print("p99 TTFT:", round(summary.p99_ttft_s * 1000, 2), "ms")
    print("goodput:", round(summary.request_goodput, 2), "req/s")


if __name__ == "__main__":
    demo()
