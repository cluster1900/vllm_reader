# dot：[TASK] #6 第 06 章复核

- Reader base: `7d4cd098cf6ec1abd8c7b570dc1df5238cf37eef`
- Source: `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`
- 文件：`book/06-paged-attention/README.md`
- 日期：2026-10-18
- 方法：固定源码、测试源码与正文/图示静态交叉核对；非真实 GPU 验证。章节保持 draft。
- 结论：已修正下列事实/图示/说明问题，不变更旧基线声明。

| 文档位置 | 问题 | 固定源码依据 | 改法 |
|---|---|---|---|
| §6.10 表及原L666–668 | “有效规模”容易把 num_actual_tokens 误读为去 padding 计数，TODO 提示缺构造/消费证据 | vllm/v1/attention/backend.py:L384–388；vllm/v1/worker/gpu_model_runner.py:L2489–2505；vllm/v1/attention/backends/flash_attn.py:L556、724、998–1008 | 明确传入 num_tokens_padded、builder 传递与 forward slicing；不删真实 TODO，不改字段名 |

## 未闭环范围

2026-10-08 全文报告中的 NVIDIA 真实生成、CUDA Graph capture/replay、NCCL/Ray、多机和性能实验仍未执行；不能以 CPU 教学模型通过替代。EPUB/网页最终重建与视觉校验应在合并后固定 Reader commit 再进行。本次不把旧报告日期或历史运行结果改写为新结果。
