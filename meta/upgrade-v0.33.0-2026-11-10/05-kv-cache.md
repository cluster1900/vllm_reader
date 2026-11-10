# 05-kv-cache：v0.33.0 升级复核（B1）

- 结论：**已更新（有实质影响）**。
- Source：`4ba787ec9959e566d3b587a796dff07673324da5`（v0.33.0）；旧正文基线 `5df63a7203cdb9328d824a27b682e77c7b9928be`（v0.32.0）；Reader base：`ed2676df`。
- 日期：2026-11-10；静态源码验证，章节保持 draft / runtime_verified=false。
- 跟踪：[bot-hub #127](https://github.com/cluster1900/bot-hub/issues/127)。
- 范围：只写 v0.33.0 祖先链里真实存在的变化（每个提交均用 `git merge-base --is-ancestor <commit> 4ba787ec9959` 核对）；不在祖先链的 main 变化仍以“未发版”注记保留，留给第二波。

## 证据与修订

| 固定源码 | 符号/入口 | 上游提交 | 结论与正文修改 |
|---|---|---|---|
| [vllm/v1/core/kv_cache_planning.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_planning.py) | `get_global_kv_cache_groups`、`get_kv_cache_configs`、`_get_layer_block_stride`、`_get_kv_cache_bytes_per_block`、`_pool_bytes_per_block`、`get_kv_cache_config_from_groups`、`DefaultKVCacheConfigBuilder` | df6c7ad03 (#61183)、b54458740 (#53558) | 规划函数从 `kv_cache_utils.py` 迁入新模块；新增可插拔 builder。5.5.3 改引用路径，新增 5.5.5 与图。 |
| [vllm/v1/core/kv_cache_config_builder.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_config_builder.py) | `KVCacheConfigBuilder`、`get_kv_cache_config_builder` | b54458740 (#53558) | 接口标注为实验性；`_get_pool_bytes_per_block` 私有且过渡。正文按“规划流程的替换点”描述，不写成稳定 API。 |
| [vllm/v1/engine/core.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/core.py) | `EngineCore._initialize_kv_caches` | 3e75ccba2 (#62330) | 全局分组前移到 profiling 之前并确定 DCP interleave；5.6.3 时序图重画，删除原“main 上的变化”中的 #62330 条目。 |
| [vllm/v1/worker/gpu_worker.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu_worker.py) | `Worker.init_device`、`determine_available_memory` | 9cede34a6 (#61044)、7128f568a (#58243) | cuBLAS handle 在初始快照前创建；MRV2 encoder graph 在 decoder 无 graph 时也估算。5.6.1 改写。 |
| [vllm/v1/core/kv_cache_utils.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_utils.py) | `get_request_block_hasher`、`generate_lookahead_block_hash_extra_keys`、`null_head_len`、`trim_null_head` | 2fb320d83 (#61589)、19b0efce3 (#61724) | lookahead 哈希与 null 头部裁剪。新增 5.12.4（含公式、符号说明、判定图）；5.9 补 null 短路。 |
| [vllm/config/vllm.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/config/vllm.py) | `num_hash_lookahead_tokens`、`adjust_dcp_kv_cache_interleave_size` | 2fb320d83 (#61589)、3e75ccba2 (#62330) | lookahead 启用条件（无 KV connector、未关闭、开 prefix caching、EAGLE 末块回退可用）。 |
| [vllm/v1/core/block_pool.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/block_pool.py) | `BlockPool.touch`、`free_blocks`、`_generate_event_extra_keys`、`_lookahead_token_ids` | 19b0efce3 (#61724)、2fb320d83 (#61589) | null 短路；`BlockStored` 事件带 `lookahead_token_ids`。 |
| [vllm/v1/core/kv_cache_coordinator.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_coordinator.py) | `drop_eagle_group_ids`、`replay_boundaries` 调用 | 2fb320d83 (#61589) | lookahead 哈希下 group 仍标 EAGLE 但不丢末块。 |
| [vllm/v1/core/kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_manager.py) | `get_computed_blocks(max_cache_hit_length)` | aea0513a3 (#62591) | 新增可选命中上限参数；5.12.2 补充。 |
| [vllm/v1/core/single_type_kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/single_type_kv_cache_manager.py) | `cache_blocks`、`MambaManager._cache_partial_tail_block`、`MambaManager.__init__` | a608b6e6e (#62206)、d561af5df (#62841)、2cc2e68d1 (#60043) | EAGLE 交汇点尾部、Mamba 重复登记、`mamba_cache_mode="all"` 删除。5.15.5 补写。 |
| [vllm/v1/kv_cache_interface.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/kv_cache_interface.py) | `KVCacheSpec.is_draft`、`SlidingWindowSpec.num_retained_tokens`、`MLAAttentionSpec.topk_buffer_width`、`get_mamba_prefill_checkpoint_position`、`get_kv_quant_mode` | c3de0b37e (#61455)、e1f314164 (#60767)、b048630a1 (#60805)、2fb320d83 (#61589)、9499c949b (#52670) | draft 层显式标签；SW 保留 token 数；checkpoint 位置按 lookahead 修正；dtype handler。 |
| [vllm/v1/worker/gpu/block_table.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/block_table.py) | `BlockTables.gather_block_tables` | e1f314164 (#60767) | 新增 5.16.1：窗口以下整块在 forward block table 中置 null（公式与算例）。ch06 批次需要同步这一处。 |

## 不修改正文的 v0.33.0 变化

- `KVQuantMode` 新增的 Octave 模式与 `CUSTOM` 取值（#59774、#52670）：只在 5.2.2 提到 dtype handler 机制，不展开各量化格式。
- `Worker.sleep` / `wake_up` / `discard` 对 KV connector 的注销与恢复（#59625）：本章只讲到 connector 接口边界。
- `local_rank` 计算改为 `ParallelConfig.get_local_gpu_index`（#61237）：属于 ch09。
- KV offload / P2P 的一批修复：正文只讲到 offload 边界，已在 5.15.4 汇总列出（均核对在祖先链中）。

## 教学示例与验证边界

`examples/ch05_kv_cache_blocks.py` 是固定完整 block 的教学模型，不模拟 lookahead 哈希、null 头部裁剪或 Mamba checkpoint；代码无影响，仍重跑测试。5.12.4、5.15.5、5.16.1 的数字算例都是对源码规则的手算，不是 GPU 运行结果；`tests/v1/worker/test_gpu_block_table.py` 的相关用例需要 CUDA 设备，本机未运行。
