# Chapter 09 source audit: upgrade

Immutable original: book/09-distributed/README.md. Evidence pinned to db9527a46873454610df6dbedf79a36d6bf1a7f6. All original line numbers refer to immutable reader, not edited copies.

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

## Target upgrade impact: substantive changes required
- Original :150,237–240,911 and exercise :1530–1531 asserted PCP+DP is universally forbidden. Target config/parallel.py:561–592 removed that check; legacy :547–552 retained it. Rewrote prose/table/exercise while keeping MRV2 PP prohibition. Updated upgrade-only examples/ch09_parallel_topology.py to remove obsolete guard and its unit test to check worker count plus DP/PCP/EP groups. This simulator still does not validate hardware/model legality.
- Original :901–914 described only long-prefill splitting, banned speculation and full graphs. Target worker/gpu/pcp_manager.py:121–197 supports DSpark/single-module MTP, constrains MLA drafts under DSpark, requires sparse MLA plus ag_rs for PCP+DCP, allows decode-separated full graphs, retains sparse MLA DCP=1 NONE restriction. Replaced the complete constraint block, not an appended disclaimer.
- Target pcp_manager.py:222–269 replicates short prefills under DCP when (2*PCP-1)*ceil(query_len/(2*PCP)) >= query_len. Updated prose AND PCP Mermaid branching; retained exact long 16-token example. Tests/v1/worker/test_gpu_pcp_manager.py:67–93 and :384–404 corroborate new full-graph and replication cases (read, not executed).
- Original :751–761 described coordination unconditionally. Target worker/dp_utils.py:22–36,240–252 adds explicit skip context. Qualified the normal synchronization contract and documented bypass output semantics.
- Header and final baseline now v0.31.0/db9527a4, reviewed 2026-10-18. Runtime remains false.

## Source-map changes
- Remove def get_kv_cache_shard_count from vllm/v1/worker/cp_utils.py: target no longer defines it. Keep check_attention_cp_compatibility; add def get_dcp_dummy_context_len and def prepare_dcp_dummy_context_metadata (target lines 60 and 77) for current context helper navigation, not as one-for-one semantic replacements.
- Add def replicated_requests to vllm/v1/worker/gpu/pcp_manager.py. Existing class PCPManager/validate_config remain.
- Consider adding target tests test_validate_config_accepts_decode_only_full_graphs and test_dcp_replicates_prefills_too_short_to_split. The JSON and Markdown maps are updated in this review package.
