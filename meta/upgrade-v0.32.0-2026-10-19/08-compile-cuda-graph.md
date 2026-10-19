# 08-compile-cuda-graph：v0.32.0 升级复核

- 结论：**已更新（有实质影响）**。
- Source: `5df63a7203cdb9328d824a27b682e77c7b9928be`；Reader base: `5f78d105690c1968885abb8c0264cd98fb7b9a18`。
- 日期：2026-10-19；静态源码验证，章节保持 draft / runtime_verified=false。

## 证据与修订

| 固定源码 | 符号/入口 | 结论与正文修改 |
|---|---|---|
| [vllm/config/vllm.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/vllm.py) | `_disable_cudagraphs_for_v2_stock_torch_compile / use_cumem_cudagraph_pool` | MRV2 stock compile 强制 graph NONE/清空 capture sizes；cuMem sleep graph pool 受多个条件门控。 |
| [vllm/config/compilation.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/compilation.py) | `CompilationConfig` | torch.compile SP/AsyncTP 相关 pass 配置和 gating 移除，不再作为当前回退分支。 |
| [vllm/config/parallel.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/parallel.py) | `use_sequence_parallel_moe` | 模型级 MoE SP 仍在，与移除 compile passes 区分。 |
| [vllm/v1/worker/gpu_model_runner.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu_model_runner.py) | `_determine_batch_execution_and_padding` | cascade 条件已移除，图示/文字不再将其列为当前 FULL graph 禁用条件。 |
| [vllm/v1/worker/gpu/cudagraph_utils.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/cudagraph_utils.py) | `profile_cudagraph_memory` | profiling 临时 pool 与真实 replay 状态分离，完成后恢复原池；不构成实际 CUDA capture 验证。 |

## 教学示例与验证边界

ch08 模型只演示 overlap、descriptor 与稳定地址契约，不模拟已删除 pass/cascade、cuMem 或 stock compile，因此代码无影响；真实 compile/capture/replay 未执行。

统一实际测试结果及逐文件变更覆盖见[总报告](README.md)。旧实验报告不作本次运行证明。
