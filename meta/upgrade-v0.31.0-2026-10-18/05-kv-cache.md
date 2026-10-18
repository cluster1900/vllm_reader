# dot：[TASK] #7 第 05 章复核

- Reader base: `7d4cd098cf6ec1abd8c7b570dc1df5238cf37eef`
- Source: `db9527a46873454610df6dbedf79a36d6bf1a7f6`
- 文件：`book/05-kv-cache/README.md`
- 日期：2026-10-18
- 方法：固定源码、测试源码与正文/图示静态交叉核对；非真实 GPU 验证。章节保持 draft。
- 升级结论：已更新；不是整章“无影响”。独立于 #6 patch，允许单独审阅。

| 文档位置 | 更新内容 | v0.31.0 源码依据 |
|---|---|---|
| §5.11、5.14–5.17 | 零 new 可附着本地 computed；多 pool reset 非原子；prefix_cacheable/DCP group、Mamba replay checkpoint 与 HiSparse 私有 host pool 边界 | vllm/v1/core/kv_cache_manager.py:L181–202、467–493、669–687；vllm/v1/core/kv_cache_coordinator.py:L319–385、691–720；vllm/v1/core/single_type_kv_cache_manager.py:L2239–2280；vllm/v1/kv_cache_interface.py:KVCacheSpec |

## 映射文件差异清单

以下是同 SHA 内容比较，不以“有 diff”代替语义结论；正文受影响点见上表。

| 文件 | 两版内容 |
|---|---|
| `vllm/v1/kv_cache_interface.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/worker/gpu_worker.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/core/kv_cache_utils.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/core/kv_cache_manager.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/core/kv_cache_coordinator.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/core/block_pool.py` | 有变化；已纳入本章差异材料 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | 有变化；已纳入本章差异材料 |
| `tests/v1/core/test_single_type_kv_cache_manager.py` | 有变化；已纳入本章差异材料 |
| `tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py` | 有变化；已纳入本章差异材料 |
| `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py` | 有变化；已纳入本章差异材料 |
| `tests/v1/core/test_reset_prefix_cache_e2e.py` | 完全相同，文件级无变化 |


## 未闭环范围

2026-10-08 全文报告中的 NVIDIA 真实生成、CUDA Graph capture/replay、NCCL/Ray、多机和性能实验仍未执行；不能以 CPU 教学模型通过替代。EPUB/网页最终重建与视觉校验应在合并后固定 Reader commit 再进行。本次不把旧报告日期或历史运行结果改写为新结果。
