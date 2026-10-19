# 05-kv-cache：v0.32.0 升级复核

- 结论：**已更新（有实质影响）**。
- Source: `5df63a7203cdb9328d824a27b682e77c7b9928be`；Reader base: `5f78d105690c1968885abb8c0264cd98fb7b9a18`。
- 日期：2026-10-19；静态源码验证，章节保持 draft / runtime_verified=false。

## 证据与修订

| 固定源码 | 符号/入口 | 结论与正文修改 |
|---|---|---|
| [vllm/v1/core/kv_cache_coordinator.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_coordinator.py) | `replay_boundaries / cache_blocks / get_num_cacheable_tokens` | 旧 get_replay_boundaries 方法删除，改用模块纯函数；Hybrid manager 自行处理 cacheable 对齐/EAGLE proof 边界。 |
| [vllm/v1/kv_cache_interface.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_cache_interface.py) | `get_mamba_prefill_checkpoint_position / KVCacheSpec` | prefix_cacheable 与 dcp_sharded 分开；EAGLE checkpoint 先按 prompt proof limit 再退一对齐单位，不能多退一个边界。 |
| [vllm/v1/engine/core.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/core.py) | `_initialize_kv_caches` | 显式传递 hash_block_size/cache_hit_alignment_tokens，不把 hash 粒度当所有 cache hit 边界。 |
| [tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py) | `test_resolved_cache_hit_alignment_matches_coordinator` | 含/不含 Sliding Window group 的 cache-hit alignment 与 coordinator 一致性证据。 |
| [vllm/v1/worker/utils.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/utils.py) | `request_memory` | ceil(total*util) 减去 loader 报告的外部权重后校验预算；普通加载无外部权重扣项。 |

## 教学示例与验证边界

ch05 教学程序为固定完整 block 模型，不模拟 Mamba/EAGLE checkpoint、HiSparse 或外部权重加载；代码无影响，仍重跑测试。纯函数算例与 GPU 实验明确分开。

统一实际测试结果及逐文件变更覆盖见[总报告](README.md)。旧实验报告不作本次运行证明。
