# dot：[TASK] #7 第 02 章复核

- Reader base: `7d4cd098cf6ec1abd8c7b570dc1df5238cf37eef`
- Source: `db9527a46873454610df6dbedf79a36d6bf1a7f6`
- 文件：`book/02-vllm-architecture/README.md`
- 日期：2026-10-18
- 方法：固定源码、测试源码与正文/图示静态交叉核对；非真实 GPU 验证。章节保持 draft。
- 升级结论：已更新；不是整章“无影响”。独立于 #6 patch，允许单独审阅。

| 文档位置 | 更新内容 | v0.31.0 源码依据 |
|---|---|---|
| §fan-out、admission、请求结构 | 先注册全部 child 再 await 发送；共享 request counters 不等于 token counters 全局共享；增加 kv_hints 边界 | vllm/v1/engine/async_llm.py:L150–172、374–535；vllm/v1/engine/admission_control.py:L13 起；vllm/v1/engine/output_processor.py:OutputProcessor.get_num_queued_tokens |

## 映射文件差异清单

以下是同 SHA 内容比较，不以“有 diff”代替语义结论；正文受影响点见上表。

| 文件 | 两版内容 |
|---|---|
| `vllm/entrypoints/llm.py` | 有变化；已纳入本章差异材料 |
| `vllm/entrypoints/offline_utils.py` | 有变化；已纳入本章差异材料 |
| `vllm/entrypoints/openai/chat_completion/serving.py` | 有变化；已纳入本章差异材料 |
| `vllm/renderers/base.py` | 有变化；已纳入本章差异材料 |
| `vllm/renderers/online_renderer.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/engine/async_llm.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/engine/input_processor.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/engine/core_client.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/engine/core.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/engine/output_processor.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/request.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/executor/uniproc_executor.py` | 完全相同，文件级无变化 |
| `tests/v1/engine/test_output_processor.py` | 有变化；已纳入本章差异材料 |
| `tests/v1/engine/test_async_llm.py` | 有变化；已纳入本章差异材料 |
| `tests/v1/engine/test_abort_final_step.py` | 有变化；已纳入本章差异材料 |


## 未闭环范围

2026-10-08 全文报告中的 NVIDIA 真实生成、CUDA Graph capture/replay、NCCL/Ray、多机和性能实验仍未执行；不能以 CPU 教学模型通过替代。EPUB/网页最终重建与视觉校验应在合并后固定 Reader commit 再进行。本次不把旧报告日期或历史运行结果改写为新结果。
