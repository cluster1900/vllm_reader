# Chapter 09 source audit: legacy

Immutable original: book/09-distributed/README.md. Evidence pinned to 5893426b88f7b3cd21101d194eb1c6f0a6f0e27b. All original line numbers refer to immutable reader, not edited copies.

## Legacy finding retained in both editions
- Original book/09-distributed/README.md:538–549 says PP remainders go to stages “靠中间”. That description hides the exact asymmetric rule. Source vllm/distributed/utils.py:127–174 (legacy), :128–175 (target) iterates range(2, remaining_layers+2), incrementing partitions[-i]. Fixed §9.6.1 to state reverse order starting at the penultimate stage, added 12/5 counterexample [2,2,3,3,2], and documented the explicit partition override. The existing 10/3 diagram remains correct (3,4,3).

## Reviewed and unchanged contracts
- World size/launcher caveat: legacy config/parallel.py:867 onward, target :899–907 and :597–599. PCP multiplies workers, DCP does not; external launcher folds DP into world_size. Original §9.3 remains valid except target PCP+DP prohibition.
- Rank coordinate/group diagrams: target distributed/parallel_state.py:2054–2065,2135–2225 retain ExternalDP×DP×PP×PCP×TP and TP/PCP/PP/DP/EP construction. Added Engram-specific groups do not invalidate the ordinary topology taught here.
- TP linear and bias semantics, PP receive/send and output-rank caveats checked against layers/linear.py, models/llama.py, gpu_worker.py, multiproc_executor.py. No conceptual rewrite required.
- DP routing score retained (legacy engine/core_client.py:1556–1590; target same algorithm). Coordinated wave diagram remains wave-level, not a fictitious per-step coordinator command.
- Expert placement remainder goes to first ranks despite stale source docstring. FusedMoE expert_map_manager.py actual quotient/remainder code and config.py EP flattening remain authoritative.
- LSE merge mathematics and post-attention A2A semantics retained. Target attention/ops/dcp.py introduces kernel-wrapper/warmup refactors and empty-shard safeguards, not arithmetic averaging.

## Verification boundary
44 chapter 09+10 teaching unit tests passed in this edition. AST-extracted upstream get_pp_indices executed for both revisions and returned [(0,2),(2,4),(4,7),(7,10),(10,12)] for 12 layers/5 stages. Chapter Markdown fence/whitespace and pedagogy checks passed. Topology, TP arithmetic and LSE teaching tests are CPU-only; no GPU, NCCL, Ray or multi-node runtime claim. Diagram data/conditions inspected; full renderer verification belongs to the publication build.

## Upgrade decision
Target changes are intentionally NOT backported into this legacy edition. In chapter 09 the original PCP+DP/speculation/full-graph restrictions are correct for its pinned source. In chapter 10 the original chat usage-event behavior and Proton eager restriction are correct for its pinned source.
