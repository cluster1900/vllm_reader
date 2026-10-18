# dot：[TASK] #6 第 04 章复核

- Reader base: `7d4cd098cf6ec1abd8c7b570dc1df5238cf37eef`
- Source: `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`
- 文件：`book/04-scheduler/README.md`
- 日期：2026-10-18
- 方法：固定源码、测试源码与正文/图示静态交叉核对；非真实 GPU 验证。章节保持 draft。
- 结论：重点复核未发现新增独立问题，正文不变。

| 文档位置 | 问题 | 固定源码依据 | 改法 |
|---|---|---|---|
| §4.5、4.8–4.20 | 本轮重点路径复核未发现需要改动的独立旧基线问题；保留原文，不制造空补丁 | vllm/v1/request.py:Request.num_tokens_with_spec；vllm/v1/core/sched/scheduler.py:L535–1140、1452–1518；tests/v1/core/test_scheduler.py:test_schedule_concurrent_partial_requests | 已核对差额表达式、running/waiting 分配失败差异、抢占保留 token、stale 输出与 fence 边界；未运行 GPU |

## 未闭环范围

2026-10-08 全文报告中的 NVIDIA 真实生成、CUDA Graph capture/replay、NCCL/Ray、多机和性能实验仍未执行；不能以 CPU 教学模型通过替代。EPUB/网页最终重建与视觉校验应在合并后固定 Reader commit 再进行。本次不把旧报告日期或历史运行结果改写为新结果。
