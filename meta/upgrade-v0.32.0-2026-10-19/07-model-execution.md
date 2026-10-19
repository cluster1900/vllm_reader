# 07-model-execution：v0.32.0 升级复核

- 结论：**已更新（有实质影响）**。
- Source: `5df63a7203cdb9328d824a27b682e77c7b9928be`；Reader base: `5f78d105690c1968885abb8c0264cd98fb7b9a18`。
- 日期：2026-10-19；静态源码验证，章节保持 draft / runtime_verified=false。

## 证据与修订

| 固定源码 | 符号/入口 | 结论与正文修改 |
|---|---|---|
| [vllm/config/vllm.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/vllm.py) | `use_v2_model_runner / _get_v2_model_runner_unsupported_features` | 移除 ROCm 模型特例、stock compile/ngram/Elastic EP 统一 fallback；保留 external-launcher+PP 等限制，新增异词表草稿限制。 |
| [vllm/v1/worker/gpu_worker.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu_worker.py) | `_make_model_runner / _init_workspace_and_model_runner / init_device` | Runner 构造拆分，查询外部权重内存后取快照，encoder-only 路径另选模型。 |
| [vllm/v1/worker/utils.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/utils.py) | `request_memory / AttentionGroup` | 外部权重预算扣除；manager/kernel table 在 metadata 边界映射。 |
| [vllm/v1/worker/gpu/states.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/states.py) | `RequestState` | n-gram speculation 选择 dense GPU all_token_ids，而非一概 UVA。 |
| [vllm/v1/worker/gpu/model_runner.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/model_runner.py) | `execute_model / sample_tokens` | 每轮保存 draft 数、DP state；非末 PP rank 配合 PCP 恢复 global batch，不能使用局部行同步输出。 |

## 教学示例与验证边界

ch07 教学模型不选择真实 MRV1/MRV2、loader、GPU tensor 或 speculative kernel，代码无影响；实际重跑测试。真实模型加载、PP/NCCL、D2H trace 未执行。

统一实际测试结果及逐文件变更覆盖见[总报告](README.md)。旧实验报告不作本次运行证明。
