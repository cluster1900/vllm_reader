# Chapter 10 source audit: legacy

Immutable original: book/10-experiments/README.md. Evidence pinned to 5893426b88f7b3cd21101d194eb1c6f0a6f0e27b. All original line numbers refer to immutable reader, not edited copies.

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

## Upgrade decision
Target changes are intentionally NOT backported into this legacy edition. In chapter 09 the original PCP+DP/speculation/full-graph restrictions are correct for its pinned source. In chapter 10 the original chat usage-event behavior and Proton eager restriction are correct for its pinned source.
