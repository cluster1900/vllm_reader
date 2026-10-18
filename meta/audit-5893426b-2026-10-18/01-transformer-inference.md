# dot：[TASK] #6 第 01 章复核

- Reader base: `7d4cd098cf6ec1abd8c7b570dc1df5238cf37eef`
- Source: `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`
- 文件：`book/01-transformer-inference/README.md`
- 日期：2026-10-18
- 方法：固定源码、测试源码与正文/图示静态交叉核对；非真实 GPU 验证。章节保持 draft。
- 结论：已修正下列事实/图示/说明问题，不变更旧基线声明。

| 文档位置 | 问题 | 固定源码依据 | 改法 |
|---|---|---|---|
| §Penalty/Bias 图，原L592–608 | 图遗漏可选 thinking-budget 掩码；greedy 也须受此约束 | vllm/v1/sample/sampler.py:L373–417 | 补 mask 节点和说明 |
| §源码链，原L1118 | LlamaForCausalLM.forward 被限定为只返回 hidden states，漏非最后 PP rank | vllm/model_executor/models/llama.py:L406–441、534–552 | 明确 Tensor / IntermediateTensors 分支 |

## 未闭环范围

2026-10-08 全文报告中的 NVIDIA 真实生成、CUDA Graph capture/replay、NCCL/Ray、多机和性能实验仍未执行；不能以 CPU 教学模型通过替代。EPUB/网页最终重建与视觉校验应在合并后固定 Reader commit 再进行。本次不把旧报告日期或历史运行结果改写为新结果。
