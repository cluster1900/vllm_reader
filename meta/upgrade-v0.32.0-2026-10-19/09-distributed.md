# 09-distributed：v0.32.0 升级复核

- 结论：**已更新（有实质影响）**。
- Source: `5df63a7203cdb9328d824a27b682e77c7b9928be`；Reader base: `5f78d105690c1968885abb8c0264cd98fb7b9a18`。
- 日期：2026-10-19；静态源码验证，章节保持 draft / runtime_verified=false。

## 证据与修订

| 固定源码 | 符号/入口 | 结论与正文修改 |
|---|---|---|
| [vllm/v1/worker/gpu/pcp_manager.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/pcp_manager.py) | `validate_config / _iter_rank_chunks` | PCP+PP 禁令删除，仍校验 MLA、encoder-decoder、MM、LoRA、spec、DCP 和 graph；PCP-only decode 按有效 decode 序号分配 owner。 |
| [vllm/config/parallel.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/parallel.py) | `pcp_shard_decode_requests / use_sequence_parallel_moe` | DCP=1 才使用 PCP decode sharding；DCP 查询仍需参与 rank 复制，模型级 SP 保留。 |
| [vllm/v1/worker/gpu/model_runner.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/model_runner.py) | `sample_tokens` | PCP+PP 非末 rank 恢复 global batch 后同步输出，不能混同本地 batch 行。 |
| [vllm/v1/engine/core_client.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/core_client.py) | `DPLBAsyncMPClient._eep_commit_pause_mode` | MRV2 EEP 无法复用 fused MoE kernel 时先 wait drain，否则 keep；取消统一 fallback 不等于任意组合无限制。 |

## 教学示例与验证边界

ch09 新增 pcp_decode_owners 和测试，覆盖 prefill/零 token 行不占 decode ordinal、DCP query 复制、PCP+PP 形式拓扑；真实 collective、KV 恢复和 PP 同步未执行。

统一实际测试结果及逐文件变更覆盖见[总报告](README.md)。旧实验报告不作本次运行证明。
