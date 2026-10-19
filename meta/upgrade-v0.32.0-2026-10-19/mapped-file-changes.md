# 章节映射文件差异覆盖

来自 [bot-hub #20 输入材料](https://github.com/cluster1900/bot-hub/tree/main/materials/20)；
此表是 v0.31.0→v0.32.0 差异清单，不把“文件改变”自动等同于“章节所有结论改变”。
正文影响与无影响范围见各章报告。新版读取文件的精确 blob 见 source-manifest.json。

## 文件明细

| 章 | 文件 | 变化 |
|---|---|---|
| 01-transformer-inference | `vllm/v1/kv_cache_interface.py` | 1 file changed, 39 insertions(+), 44 deletions(-) |
| 01-transformer-inference | `vllm/model_executor/models/llama.py` | 无变化 |
| 01-transformer-inference | `vllm/model_executor/layers/attention/attention.py` | 无变化 |
| 01-transformer-inference | `vllm/model_executor/layers/logits_processor.py` | 无变化 |
| 01-transformer-inference | `vllm/sampling_params.py` | 1 file changed, 134 insertions(+), 35 deletions(-) |
| 01-transformer-inference | `vllm/v1/sample/sampler.py` | 1 file changed, 3 insertions(+) |
| 01-transformer-inference | `vllm/v1/worker/gpu/sample/sampler.py` | 1 file changed, 4 insertions(+), 1 deletion(-) |
| 01-transformer-inference | `tests/test_sampling_params.py` | 1 file changed, 3 insertions(+), 7 deletions(-) |
| 01-transformer-inference | `tests/v1/sample/test_topk_topp_sampler.py` | 1 file changed, 11 insertions(+), 2 deletions(-) |
| 01-transformer-inference | `tests/kernels/core/test_rotary_embedding.py` | 无变化 |
| 01-transformer-inference | `vllm/v1/worker/gpu/sample/logits_processor/interface.py` | 无变化 |
| 01-transformer-inference | `vllm/v1/engine/input_processor.py` | 1 file changed, 57 insertions(+), 36 deletions(-) |
| 02-vllm-architecture | `vllm/entrypoints/llm.py` | 1 file changed, 54 insertions(+), 78 deletions(-) |
| 02-vllm-architecture | `vllm/entrypoints/offline_utils.py` | 1 file changed, 634 deletions(-) |
| 02-vllm-architecture | `vllm/entrypoints/openai/chat_completion/serving.py` | 1 file changed, 37 insertions(+), 35 deletions(-) |
| 02-vllm-architecture | `vllm/renderers/base.py` | 1 file changed, 66 insertions(+), 4 deletions(-) |
| 02-vllm-architecture | `vllm/renderers/online_renderer.py` | 1 file changed, 61 insertions(+), 18 deletions(-) |
| 02-vllm-architecture | `vllm/v1/engine/async_llm.py` | 1 file changed, 111 insertions(+), 46 deletions(-) |
| 02-vllm-architecture | `vllm/v1/engine/input_processor.py` | 1 file changed, 57 insertions(+), 36 deletions(-) |
| 02-vllm-architecture | `vllm/v1/engine/core_client.py` | 1 file changed, 128 insertions(+), 61 deletions(-) |
| 02-vllm-architecture | `vllm/v1/engine/core.py` | 1 file changed, 98 insertions(+), 21 deletions(-) |
| 02-vllm-architecture | `vllm/v1/engine/output_processor.py` | 1 file changed, 85 insertions(+), 7 deletions(-) |
| 02-vllm-architecture | `vllm/v1/request.py` | 无变化 |
| 02-vllm-architecture | `vllm/v1/executor/uniproc_executor.py` | 无变化 |
| 02-vllm-architecture | `tests/v1/engine/test_output_processor.py` | 1 file changed, 324 insertions(+), 1 deletion(-) |
| 02-vllm-architecture | `tests/v1/engine/test_async_llm.py` | 1 file changed, 283 insertions(+), 9 deletions(-) |
| 02-vllm-architecture | `tests/v1/engine/test_abort_final_step.py` | 无变化 |
| 02-vllm-architecture | `vllm/v1/engine/admission_control.py` | 无变化 |
| 03-engine-core | `vllm/engine/arg_utils.py` | 1 file changed, 84 insertions(+), 19 deletions(-) |
| 03-engine-core | `vllm/config/vllm.py` | 1 file changed, 186 insertions(+), 288 deletions(-) |
| 03-engine-core | `vllm/v1/engine/llm_engine.py` | 1 file changed, 49 insertions(+), 3 deletions(-) |
| 03-engine-core | `vllm/v1/engine/async_llm.py` | 1 file changed, 111 insertions(+), 46 deletions(-) |
| 03-engine-core | `vllm/v1/engine/core_client.py` | 1 file changed, 128 insertions(+), 61 deletions(-) |
| 03-engine-core | `vllm/v1/engine/core.py` | 1 file changed, 98 insertions(+), 21 deletions(-) |
| 03-engine-core | `vllm/v1/engine/__init__.py` | 1 file changed, 3 insertions(+) |
| 03-engine-core | `tests/v1/engine/test_engine_core.py` | 无变化 |
| 03-engine-core | `tests/v1/engine/test_abort_final_step.py` | 无变化 |
| 03-engine-core | `tests/v1/engine/test_startup_watch_processes.py` | 1 file changed, 2 insertions(+), 2 deletions(-) |
| 04-scheduler | `vllm/v1/core/sched/scheduler.py` | 1 file changed, 217 insertions(+), 140 deletions(-) |
| 04-scheduler | `vllm/v1/core/sched/output.py` | 1 file changed, 10 insertions(+), 8 deletions(-) |
| 04-scheduler | `vllm/v1/core/sched/request_queue.py` | 无变化 |
| 04-scheduler | `vllm/v1/request.py` | 无变化 |
| 04-scheduler | `vllm/config/scheduler.py` | 1 file changed, 2 insertions(+), 3 deletions(-) |
| 04-scheduler | `vllm/v1/core/kv_cache_manager.py` | 1 file changed, 7 insertions(+), 47 deletions(-) |
| 04-scheduler | `vllm/v1/core/sched/async_scheduler.py` | 无变化 |
| 04-scheduler | `tests/v1/core/test_scheduler.py` | 1 file changed, 418 insertions(+), 66 deletions(-) |
| 04-scheduler | `tests/v1/core/test_deferred_block_free.py` | 无变化 |
| 05-kv-cache | `vllm/v1/kv_cache_interface.py` | 1 file changed, 39 insertions(+), 44 deletions(-) |
| 05-kv-cache | `vllm/v1/worker/gpu_worker.py` | 1 file changed, 134 insertions(+), 70 deletions(-) |
| 05-kv-cache | `vllm/v1/core/kv_cache_utils.py` | 1 file changed, 202 insertions(+), 353 deletions(-) |
| 05-kv-cache | `vllm/v1/core/kv_cache_manager.py` | 1 file changed, 7 insertions(+), 47 deletions(-) |
| 05-kv-cache | `vllm/v1/core/kv_cache_coordinator.py` | 1 file changed, 90 insertions(+), 124 deletions(-) |
| 05-kv-cache | `vllm/v1/core/block_pool.py` | 1 file changed, 3 insertions(+), 6 deletions(-) |
| 05-kv-cache | `vllm/v1/core/single_type_kv_cache_manager.py` | 1 file changed, 105 insertions(+), 126 deletions(-) |
| 05-kv-cache | `tests/v1/core/test_single_type_kv_cache_manager.py` | 1 file changed, 99 insertions(+), 1 deletion(-) |
| 05-kv-cache | `tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py` | 无变化 |
| 05-kv-cache | `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py` | 1 file changed, 204 insertions(+), 57 deletions(-) |
| 05-kv-cache | `tests/v1/core/test_reset_prefix_cache_e2e.py` | 无变化 |
| 06-paged-attention | `vllm/v1/worker/gpu/buffer_utils.py` | 1 file changed, 3 insertions(+), 3 deletions(-) |
| 06-paged-attention | `vllm/model_executor/layers/attention/attention.py` | 无变化 |
| 06-paged-attention | `vllm/v1/attention/backend.py` | 1 file changed, 20 insertions(+), 43 deletions(-) |
| 06-paged-attention | `vllm/v1/attention/selector.py` | 无变化 |
| 06-paged-attention | `vllm/v1/attention/backends/registry.py` | 1 file changed, 87 insertions(+), 2 deletions(-) |
| 06-paged-attention | `vllm/platforms/cuda.py` | 1 file changed, 20 deletions(-) |
| 06-paged-attention | `vllm/v1/worker/block_table.py` | 1 file changed, 10 insertions(+), 119 deletions(-) |
| 06-paged-attention | `vllm/v1/worker/gpu/block_table.py` | 1 file changed, 9 insertions(+), 22 deletions(-) |
| 06-paged-attention | `vllm/v1/worker/gpu_model_runner.py` | 1 file changed, 127 insertions(+), 403 deletions(-) |
| 06-paged-attention | `vllm/v1/worker/gpu/model_runner.py` | 1 file changed, 137 insertions(+), 100 deletions(-) |
| 06-paged-attention | `vllm/v1/attention/backends/flash_attn.py` | 1 file changed, 149 insertions(+), 401 deletions(-) |
| 06-paged-attention | `vllm/v1/attention/ops/paged_attn.py` | 无变化 |
| 06-paged-attention | `vllm/v1/attention/backends/rocm_attn.py` | 1 file changed, 5 insertions(+), 39 deletions(-) |
| 06-paged-attention | `tests/v1/worker/test_gpu_block_table.py` | 1 file changed, 11 insertions(+), 20 deletions(-) |
| 06-paged-attention | `tests/v1/attention/test_backend_per_kind.py` | 无变化 |
| 06-paged-attention | `tests/v1/attention/test_cuda_backend_probe_errors.py` | 无变化 |
| 07-model-execution | `vllm/envs.py` | 1 file changed, 74 insertions(+), 16 deletions(-) |
| 07-model-execution | `vllm/v1/executor/abstract.py` | 1 file changed, 37 insertions(+), 2 deletions(-) |
| 07-model-execution | `vllm/v1/executor/uniproc_executor.py` | 无变化 |
| 07-model-execution | `vllm/v1/executor/multiproc_executor.py` | 1 file changed, 6 insertions(+), 1 deletion(-) |
| 07-model-execution | `vllm/v1/worker/worker_base.py` | 1 file changed, 7 insertions(+) |
| 07-model-execution | `vllm/v1/worker/gpu_worker.py` | 1 file changed, 134 insertions(+), 70 deletions(-) |
| 07-model-execution | `vllm/v1/worker/gpu_model_runner.py` | 1 file changed, 127 insertions(+), 403 deletions(-) |
| 07-model-execution | `vllm/v1/worker/gpu/model_runner.py` | 1 file changed, 137 insertions(+), 100 deletions(-) |
| 07-model-execution | `vllm/v1/worker/gpu/input_batch.py` | 1 file changed, 13 insertions(+), 10 deletions(-) |
| 07-model-execution | `vllm/v1/worker/gpu/states.py` | 1 file changed, 5 insertions(+), 2 deletions(-) |
| 07-model-execution | `vllm/v1/worker/gpu/buffer_utils.py` | 1 file changed, 3 insertions(+), 3 deletions(-) |
| 07-model-execution | `vllm/model_executor/model_loader/__init__.py` | 无变化 |
| 07-model-execution | `vllm/model_executor/model_loader/utils.py` | 1 file changed, 50 insertions(+), 3 deletions(-) |
| 07-model-execution | `vllm/model_executor/model_loader/default_loader.py` | 1 file changed, 50 insertions(+), 7 deletions(-) |
| 07-model-execution | `vllm/model_executor/models/registry.py` | 1 file changed, 28 insertions(+), 9 deletions(-) |
| 07-model-execution | `vllm/model_executor/models/llama.py` | 无变化 |
| 07-model-execution | `vllm/v1/sample/sampler.py` | 1 file changed, 3 insertions(+) |
| 07-model-execution | `vllm/v1/worker/gpu/sample/sampler.py` | 1 file changed, 4 insertions(+), 1 deletion(-) |
| 07-model-execution | `vllm/v1/outputs.py` | 1 file changed, 10 insertions(+), 4 deletions(-) |
| 07-model-execution | `vllm/sequence.py` | 无变化 |
| 07-model-execution | `vllm/config/vllm.py` | 1 file changed, 186 insertions(+), 288 deletions(-) |
| 07-model-execution | `tests/distributed/test_multiproc_executor.py` | 无变化 |
| 07-model-execution | `tests/v1/worker/test_gpu_worker.py` | 1 file changed, 84 insertions(+), 5 deletions(-) |
| 07-model-execution | `tests/test_config.py` | 1 file changed, 311 insertions(+), 101 deletions(-) |
| 08-compile-cuda-graph | `vllm/envs.py` | 1 file changed, 74 insertions(+), 16 deletions(-) |
| 08-compile-cuda-graph | `vllm/config/vllm.py` | 1 file changed, 186 insertions(+), 288 deletions(-) |
| 08-compile-cuda-graph | `vllm/config/scheduler.py` | 1 file changed, 2 insertions(+), 3 deletions(-) |
| 08-compile-cuda-graph | `vllm/config/compilation.py` | 1 file changed, 35 insertions(+), 58 deletions(-) |
| 08-compile-cuda-graph | `vllm/v1/core/sched/async_scheduler.py` | 无变化 |
| 08-compile-cuda-graph | `vllm/v1/engine/core.py` | 1 file changed, 98 insertions(+), 21 deletions(-) |
| 08-compile-cuda-graph | `vllm/v1/worker/gpu/async_utils.py` | 1 file changed, 2 insertions(+), 2 deletions(-) |
| 08-compile-cuda-graph | `vllm/v1/worker/gpu/buffer_utils.py` | 1 file changed, 3 insertions(+), 3 deletions(-) |
| 08-compile-cuda-graph | `vllm/v1/worker/gpu/states.py` | 1 file changed, 5 insertions(+), 2 deletions(-) |
| 08-compile-cuda-graph | `vllm/compilation/decorators.py` | 1 file changed, 38 insertions(+), 10 deletions(-) |
| 08-compile-cuda-graph | `vllm/compilation/wrapper.py` | 1 file changed, 18 insertions(+), 1 deletion(-) |
| 08-compile-cuda-graph | `vllm/compilation/backends.py` | 1 file changed, 31 insertions(+), 15 deletions(-) |
| 08-compile-cuda-graph | `vllm/compilation/piecewise_backend.py` | 无变化 |
| 08-compile-cuda-graph | `vllm/compilation/caching.py` | 无变化 |
| 08-compile-cuda-graph | `vllm/forward_context.py` | 无变化 |
| 08-compile-cuda-graph | `vllm/v1/cudagraph_dispatcher.py` | 1 file changed, 2 insertions(+), 2 deletions(-) |
| 08-compile-cuda-graph | `vllm/compilation/cuda_graph.py` | 1 file changed, 5 insertions(+), 6 deletions(-) |
| 08-compile-cuda-graph | `vllm/v1/attention/backend.py` | 1 file changed, 20 insertions(+), 43 deletions(-) |
| 08-compile-cuda-graph | `vllm/v1/worker/gpu_worker.py` | 1 file changed, 134 insertions(+), 70 deletions(-) |
| 08-compile-cuda-graph | `vllm/v1/worker/gpu_model_runner.py` | 1 file changed, 127 insertions(+), 403 deletions(-) |
| 08-compile-cuda-graph | `vllm/v1/worker/gpu/cudagraph_utils.py` | 1 file changed, 29 insertions(+), 9 deletions(-) |
| 08-compile-cuda-graph | `vllm/v1/worker/gpu/model_runner.py` | 1 file changed, 137 insertions(+), 100 deletions(-) |
| 08-compile-cuda-graph | `tests/v1/core/test_async_scheduler.py` | 1 file changed, 1 deletion(-) |
| 08-compile-cuda-graph | `tests/v1/core/test_scheduler.py` | 1 file changed, 418 insertions(+), 66 deletions(-) |
| 08-compile-cuda-graph | `tests/v1/cudagraph/test_cudagraph_dispatch.py` | 1 file changed, 15 insertions(+), 7 deletions(-) |
| 08-compile-cuda-graph | `tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py` | 无变化 |
| 08-compile-cuda-graph | `tests/test_config.py` | 1 file changed, 311 insertions(+), 101 deletions(-) |
| 09-distributed | `vllm/config/parallel.py` | 1 file changed, 53 insertions(+), 11 deletions(-) |
| 09-distributed | `vllm/distributed/parallel_state.py` | 1 file changed, 89 insertions(+), 3 deletions(-) |
| 09-distributed | `vllm/distributed/utils.py` | 1 file changed, 10 insertions(+), 19 deletions(-) |
| 09-distributed | `vllm/model_executor/layers/linear.py` | 无变化 |
| 09-distributed | `vllm/model_executor/models/utils.py` | 1 file changed, 27 insertions(+), 4 deletions(-) |
| 09-distributed | `vllm/model_executor/models/llama.py` | 无变化 |
| 09-distributed | `vllm/v1/worker/gpu_worker.py` | 1 file changed, 134 insertions(+), 70 deletions(-) |
| 09-distributed | `vllm/v1/engine/core_client.py` | 1 file changed, 128 insertions(+), 61 deletions(-) |
| 09-distributed | `vllm/v1/engine/coordinator.py` | 无变化 |
| 09-distributed | `vllm/v1/engine/core.py` | 1 file changed, 98 insertions(+), 21 deletions(-) |
| 09-distributed | `vllm/v1/worker/dp_utils.py` | 无变化 |
| 09-distributed | `vllm/v1/executor/abstract.py` | 1 file changed, 37 insertions(+), 2 deletions(-) |
| 09-distributed | `vllm/v1/executor/multiproc_executor.py` | 1 file changed, 6 insertions(+), 1 deletion(-) |
| 09-distributed | `vllm/v1/executor/ray_executor.py` | 无变化 |
| 09-distributed | `vllm/v1/executor/ray_executor_v2.py` | 无变化 |
| 09-distributed | `vllm/v1/executor/uniproc_executor.py` | 无变化 |
| 09-distributed | `vllm/model_executor/layers/fused_moe/config.py` | 1 file changed, 46 insertions(+) |
| 09-distributed | `vllm/model_executor/layers/fused_moe/expert_map_manager.py` | 无变化 |
| 09-distributed | `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | 1 file changed, 5 insertions(+), 2 deletions(-) |
| 09-distributed | `vllm/model_executor/layers/fused_moe/modular_kernel.py` | 1 file changed, 78 insertions(+), 68 deletions(-) |
| 09-distributed | `vllm/model_executor/layers/fused_moe/all2all_utils.py` | 1 file changed, 7 insertions(+) |
| 09-distributed | `vllm/v1/worker/gpu/pcp_manager.py` | 1 file changed, 32 insertions(+), 8 deletions(-) |
| 09-distributed | `vllm/v1/worker/cp_utils.py` | 1 file changed, 2 insertions(+), 4 deletions(-) |
| 09-distributed | `vllm/v1/attention/ops/dcp.py` | 无变化 |
| 09-distributed | `vllm/v1/core/kv_cache_utils.py` | 1 file changed, 202 insertions(+), 353 deletions(-) |
| 09-distributed | `tests/distributed/test_pipeline_partition.py` | 无变化 |
| 09-distributed | `tests/distributed/test_pipeline_parallel.py` | 无变化 |
| 09-distributed | `tests/v1/engine/test_engine_core_client.py` | 1 file changed, 35 insertions(+) |
| 09-distributed | `tests/v1/distributed/test_async_llm_dp.py` | 1 file changed, 25 insertions(+), 4 deletions(-) |
| 09-distributed | `tests/distributed/test_expert_placement.py` | 无变化 |
| 09-distributed | `tests/v1/worker/test_gpu_pcp_manager.py` | 1 file changed, 221 insertions(+), 12 deletions(-) |
| 09-distributed | `tests/distributed/test_dcp_a2a.py` | 无变化 |
| 09-distributed | `tests/distributed/test_context_parallel.py` | 无变化 |
| 09-distributed | `tests/distributed/test_multiproc_executor.py` | 无变化 |
| 10-experiments | `vllm/benchmarks/latency.py` | 1 file changed, 1 insertion(+) |
| 10-experiments | `vllm/benchmarks/throughput.py` | 1 file changed, 6 insertions(+), 31 deletions(-) |
| 10-experiments | `vllm/benchmarks/serve.py` | 1 file changed, 14 insertions(+), 39 deletions(-) |
| 10-experiments | `vllm/benchmarks/lib/endpoint_request_func.py` | 1 file changed, 28 insertions(+), 2 deletions(-) |
| 10-experiments | `vllm/benchmarks/datasets/datasets.py` | 1 file changed, 13 insertions(+), 182 deletions(-) |
| 10-experiments | `vllm/benchmarks/sweep/param_sweep.py` | 无变化 |
| 10-experiments | `vllm/benchmarks/sweep/serve_workload.py` | 1 file changed, 27 insertions(+), 6 deletions(-) |
| 10-experiments | `vllm/benchmarks/sweep/plot_pareto.py` | 无变化 |
| 10-experiments | `vllm/v1/metrics/stats.py` | 1 file changed, 61 insertions(+), 6 deletions(-) |
| 10-experiments | `vllm/v1/metrics/loggers.py` | 1 file changed, 155 insertions(+), 11 deletions(-) |
| 10-experiments | `vllm/v1/metrics/prometheus.py` | 无变化 |
| 10-experiments | `vllm/v1/metrics/perf.py` | 1 file changed, 1 insertion(+), 2 deletions(-) |
| 10-experiments | `vllm/config/profiler.py` | 1 file changed, 40 insertions(+), 9 deletions(-) |
| 10-experiments | `vllm/v1/worker/gpu_worker.py` | 1 file changed, 134 insertions(+), 70 deletions(-) |
| 10-experiments | `docs/contributing/profiling.md` | 1 file changed, 37 insertions(+) |
| 10-experiments | `benchmarks/benchmark_block_pool.py` | 无变化 |
| 10-experiments | `benchmarks/benchmark_prefix_caching.py` | 无变化 |
| 10-experiments | `tests/benchmarks/test_latency_cli.py` | 无变化 |
| 10-experiments | `tests/benchmarks/test_throughput_cli.py` | 1 file changed, 3 insertions(+), 2 deletions(-) |
| 10-experiments | `tests/benchmarks/test_serve_cli.py` | 无变化 |
| 10-experiments | `tests/benchmarks/sweep/test_param_sweep.py` | 1 file changed, 134 insertions(+), 8 deletions(-) |
| 10-experiments | `vllm/profiler/wrapper.py` | 1 file changed, 104 insertions(+), 26 deletions(-) |


