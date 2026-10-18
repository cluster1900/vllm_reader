# Chapter 10 source audit: upgrade

Immutable original: book/10-experiments/README.md. Evidence pinned to db9527a46873454610df6dbedf79a36d6bf1a7f6. All original line numbers refer to immutable reader, not edited copies.

## Legacy findings retained in both editions
- Original book/10-experiments/README.md:276–279 says missing goodput thresholds yield a zero result field. Source legacy benchmarks/serve.py:584–647,739,1204–1208,1288 distinguishes internal zero from exported JSON null. Target corresponding lines 593 onward,749,1214–1218,1298 retain it. Fixed §10.3.2 to name both representations and suppressed display, avoiding interpreting an unconfigured metric as failure.
- Original :283 goodput diagram admitted “All completed requests”, omitting success prerequisite. Source calculate_metrics guards all samples by outputs[i].success (legacy :592 onward, target :601 onward). Diagram now begins Successful completed requests.
- Original :812–813 used named line syntax inside xychart-beta. Replaced with standard line [values] declarations and identified both series in adjacent text. Values remain explicitly pedagogical, not performance evidence. Full rendering is delegated to the publication build.

## Reviewed and unchanged contracts
- calculate_metrics TTFT/TPOT/ITL, single-token TPOT exclusion vs goodput zero, AND thresholds and units checked against both serve.py revisions.
- Arrival Gamma sampling, finite-sequence rescaling, semaphore exclusion from request timing, benchmark duration including scheduling/drain, ready check/warmup boundaries checked against serve.py. No upgrade formula change.
- latency.py only adds model_id to output JSON; fixed-batch timing, warmups and prefix-cache default remain. throughput.py changes dataset argument adaptation and model_id, not total request timing/token-count distinction.
- Internal clocks and queue/prefill/decode/inference definitions checked in metrics/stats.py; exported metric names and computed-prompt logging checked in metrics/loggers.py. No prose change needed for existing qualified descriptions.
- Pareto definition vs plot_pareto.py approximation, request_rate dimensional warning, and missing PCP in inferred GPU count retained; these are already correctly caveated.
- Amdahl, Little's Law boundary, benchmark/profiler separation and illustrative figures remain conceptual material, not hardware measurements.

## Verification boundary
44 chapter 09+10 teaching unit tests pass in this edition. Chapter Markdown fence/whitespace and pedagogy checks pass. CPU examples verify formulas, not real SSE transports, profiler replay, distributed execution or GPU performance. No GPU/profiler runtime results fabricated. Diagram semantic corrections inspected; Full diagram rendering remains unverified.

## Target upgrade impact: substantive changes required
- Original :226–233 says chat usage-only events advance the final clock. Target benchmarks/lib/endpoint_request_func.py:396–437 moves most_recent_timestamp into choices branch. Replaced this passage; explicitly warns lower E2EL/TPOT across versions can be a metric fix rather than GPU speedup.
- Target endpoint_request_func.py:447–619 introduces SSE Responses parsing, token/reasoning deltas and terminal/error rules. Added the adapter-specific timing distinction; usage terminal does not extend latency, absence of token delta or terminal fails the request. Main request timeline still deliberately uses “final measured event”, so remains valid.
- Original :914–938 profiler explanation became materially stale. Target profiler/wrapper.py:191,394–485,607,644–727 moves wrappers/factories, validates platform activities, and supports Proton CUDA Graph attribution. Config/profiler.py:55–61,86–89,248–263 plus docs/contributing/profiling.md:91–160 require Triton>=3.7, tree data, no periodic_flushing for attribution; Chrome trace still needs eager. Rewrote Proton subsection and platform activities, added actual factory navigation. This is not just a path substitution.
- Target latency.py:172 and throughput.py:1144 add model_id; serve.py:1549,2026 redacts header namespace. Added limited schema/provenance note without claiming total output sanitization.
- Header/footer now v0.31.0/db9527a4, reviewed 2026-10-18. Runtime remains false.

## Source-map changes
- Move class TorchProfilerWrapper, class CudaProfilerWrapper, class ProtonProfilerWrapper out of gpu_worker.py symbol list into new vllm/profiler/wrapper.py entry. Definitions target lines 191,607,394. Add def create_worker_profiler (675), def create_graph_capture_profiler (704), def validate_worker_profiler_config (644). Keep Worker/profile routing anchors in gpu_worker.py.
- Add async def async_request_openai_responses to benchmarks/lib/endpoint_request_func.py anchors (506). The JSON and Markdown maps are updated in this review package.
