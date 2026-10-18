# dot：[TASK] #7 第 06 章复核

- Reader base: `7d4cd098cf6ec1abd8c7b570dc1df5238cf37eef`
- Source: `db9527a46873454610df6dbedf79a36d6bf1a7f6`
- 文件：`book/06-paged-attention/README.md`
- 日期：2026-10-18
- 方法：固定源码、测试源码与正文/图示静态交叉核对；非真实 GPU 验证。章节保持 draft。
- 升级结论：已更新；不是整章“无影响”。独立于 #6 patch，允许单独审阅。

| 文档位置 | 更新内容 | v0.31.0 源码依据 |
|---|---|---|
| §6.6–6.7、6.10、6.15–6.16 | UVA fallback/冷 tensor 内容分支、dummy idx mask、逐 group DCP、R-SWA selector、mm-prefix 候选排序、FA4 warmup 边界；num_actual_tokens 行号迁移 | vllm/v1/worker/gpu/buffer_utils.py:L41–96、153–175、205–213；vllm/v1/worker/gpu/block_table.py:L319–364；vllm/platforms/cuda.py:L154–180；vllm/v1/attention/selector.py:L170–177；vllm/v1/attention/backend.py:L403–407；vllm/v1/worker/gpu_model_runner.py:L2424–2440；vllm/v1/attention/backends/flash_attn.py:L825、993、1286–1296 |

## 映射文件差异清单

以下是同 SHA 内容比较，不以“有 diff”代替语义结论；正文受影响点见上表。

| 文件 | 两版内容 |
|---|---|
| `vllm/v1/worker/gpu/buffer_utils.py` | 有变化；已纳入本章差异材料 |
| `vllm/model_executor/layers/attention/attention.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/attention/backend.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/attention/selector.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/attention/backends/registry.py` | 有变化；已纳入本章差异材料 |
| `vllm/platforms/cuda.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/worker/block_table.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/worker/gpu/block_table.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/worker/gpu_model_runner.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/worker/gpu/model_runner.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/attention/backends/flash_attn.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/attention/ops/paged_attn.py` | 完全相同，文件级无变化 |
| `vllm/v1/attention/backends/rocm_attn.py` | 有变化；已纳入本章差异材料 |
| `tests/v1/worker/test_gpu_block_table.py` | 有变化；已纳入本章差异材料 |
| `tests/v1/attention/test_backend_per_kind.py` | 有变化；已纳入本章差异材料 |
| `tests/v1/attention/test_cuda_backend_probe_errors.py` | 完全相同，文件级无变化 |


## 未闭环范围

2026-10-08 全文报告中的 NVIDIA 真实生成、CUDA Graph capture/replay、NCCL/Ray、多机和性能实验仍未执行；不能以 CPU 教学模型通过替代。EPUB/网页最终重建与视觉校验应在合并后固定 Reader commit 再进行。本次不把旧报告日期或历史运行结果改写为新结果。
