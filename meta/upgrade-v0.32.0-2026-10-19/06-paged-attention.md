# 06-paged-attention：v0.32.0 升级复核

- 结论：**已更新（有实质影响）**。
- Source: `5df63a7203cdb9328d824a27b682e77c7b9928be`；Reader base: `5f78d105690c1968885abb8c0264cd98fb7b9a18`。
- 日期：2026-10-19；静态源码验证，章节保持 draft / runtime_verified=false。

## 证据与修订

| 固定源码 | 符号/入口 | 结论与正文修改 |
|---|---|---|
| [vllm/v1/worker/block_table.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/block_table.py) | `BlockTable` | map_to_kernel_blocks 删除；持久表、gather、slot mapping 保持 manager 坐标。 |
| [vllm/v1/worker/utils.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/utils.py) | `AttentionGroup.map_to_kernel_block_table / map_kv_cache_to_kernel_blocks` | backend metadata 边界映射 read table 和 cache view；packed 布局按 kernel_block_stride，不总是 block-size 比。 |
| [vllm/v1/worker/gpu/attn_utils.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/attn_utils.py) | `build_attn_metadata` | MRV2 经 AttentionGroup 统一 metadata 包装。 |
| [tests/v1/worker/test_gpu_block_table.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/tests/v1/worker/test_gpu_block_table.py) | `test_dcp_slot_mapping_only_shards_sharded_groups` | 替代旧 smaller-kernel-blocks 用例，验证仅 dcp_sharded group 转换 local slot。 |
| [vllm/v1/attention/backends/flash_attn.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/attention/backends/flash_attn.py) | `FlashAttentionMetadataBuilder / FlashAttentionImpl` | cascade 计算与字段移除；分页 KV 和前缀缓存并未因此移除。 |

## 教学示例与验证边界

ch06 新增 packed kernel stride、非原地 manager→backend table 映射测试；slot 参数改为通用 block_size，写路径用 manager size。实验手算协议同步修改。没有执行上游 GPU block-table 测试。

统一实际测试结果及逐文件变更覆盖见[总报告](README.md)。旧实验报告不作本次运行证明。
