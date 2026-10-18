# dot：[TASK] #7 第 04 章复核

- Reader base: `7d4cd098cf6ec1abd8c7b570dc1df5238cf37eef`
- Source: `db9527a46873454610df6dbedf79a36d6bf1a7f6`
- 文件：`book/04-scheduler/README.md`
- 日期：2026-10-18
- 方法：固定源码、测试源码与正文/图示静态交叉核对；非真实 GPU 验证。章节保持 draft。
- 升级结论：已更新；不是整章“无影响”。独立于 #6 patch，允许单独审阅。

| 文档位置 | 更新内容 | v0.31.0 源码依据 |
|---|---|---|
| §4.6、4.7–4.8、4.10–4.11、参数表 | active seq 准入独立于 Runner capacity；单请求绕过 threshold、adaptive fair share；pending/fenced frees 不盲目抢占；保留三请求精确测试并标明教学例不等于新版单请求 trace | vllm/config/scheduler.py:L70–96、336–345；vllm/v1/core/sched/scheduler.py:L126–139、611–626、755–820、890–895；tests/v1/core/test_scheduler.py:L694–824；tests/v1/core/test_deferred_block_free.py:L249–294 |

## 映射文件差异清单

以下是同 SHA 内容比较，不以“有 diff”代替语义结论；正文受影响点见上表。

| 文件 | 两版内容 |
|---|---|
| `vllm/v1/core/sched/scheduler.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/core/sched/output.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/core/sched/request_queue.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/request.py` | 有变化；已纳入本章差异材料 |
| `vllm/config/scheduler.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/core/kv_cache_manager.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/core/sched/async_scheduler.py` | 有变化；已纳入本章差异材料 |
| `tests/v1/core/test_scheduler.py` | 有变化；已纳入本章差异材料 |
| `tests/v1/core/test_deferred_block_free.py` | 有变化；已纳入本章差异材料 |


## 未闭环范围

2026-10-08 全文报告中的 NVIDIA 真实生成、CUDA Graph capture/replay、NCCL/Ray、多机和性能实验仍未执行；不能以 CPU 教学模型通过替代。EPUB/网页最终重建与视觉校验应在合并后固定 Reader commit 再进行。本次不把旧报告日期或历史运行结果改写为新结果。
