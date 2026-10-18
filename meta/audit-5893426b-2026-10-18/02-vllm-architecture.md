# dot：[TASK] #6 第 02 章复核

- Reader base: `7d4cd098cf6ec1abd8c7b570dc1df5238cf37eef`
- Source: `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`
- 文件：`book/02-vllm-architecture/README.md`
- 日期：2026-10-18
- 方法：固定源码、测试源码与正文/图示静态交叉核对；非真实 GPU 验证。章节保持 draft。
- 结论：已修正下列事实/图示/说明问题，不变更旧基线声明。

| 文档位置 | 问题 | 固定源码依据 | 改法 |
|---|---|---|---|
| §busy loop 图，原L980–984 | active 分支直接 step_fn，漏处理已到达的 ADD/ABORT/UTILITY | vllm/v1/engine/core.py:L1422–1477 | 图中补非阻塞 input_queue drain，正文说明忙碌不等于不接单 |

## 未闭环范围

2026-10-08 全文报告中的 NVIDIA 真实生成、CUDA Graph capture/replay、NCCL/Ray、多机和性能实验仍未执行；不能以 CPU 教学模型通过替代。EPUB/网页最终重建与视觉校验应在合并后固定 Reader commit 再进行。本次不把旧报告日期或历史运行结果改写为新结果。
