# 全书再次源码与教学复核

本轮以先前全书审阅为基础，再核对 10 章的版本、主调用链、边界条件、引用与教学示例。
受众仍为初中级程序员和软件工程学生：先建立现象、例子和状态账本，再进入实现与可选分支。

## 版本与验证范围

- Source: `vllm-project/vllm`，`5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`。
- Source branch / dirty: `main` / `false`，上游只读。
- Reader base: `48c6e74e1216180cb9a49267c4f5ffaf5c98c631` 加本次变更；最终提交见 Git 历史。
- Date: 2026-09-08。
- Environment: Apple Silicon arm64 / macOS / Python 3.14。
- Method: 固定 revision 的静态调用链与测试阅读、Python AST 定义核对、CPU 教学运行、EPUB 内容比较。
- Limit: 未运行真实 vLLM 模型、NVIDIA CUDA、NCCL/Ray 多卡或性能 benchmark；章节继续保持 draft。

## 逐章覆盖与本轮修复

| 章 | 复核链与结果 |
|---|---|
| 01 推理基础 | 重查 Llama QKV/RoPE、LM head、MRV1/MRV2 sampling。补明 top-k 边界并列行为和 presence/frequency 仅计输出；不再要求读者预先熟悉产品术语。 |
| 02 请求生命周期 | 重查 offline 请求编号、排序及输入/输出处理。修复教学程序按外部 ID 排序、固定响应计划提前截断、最后一个 abort 未交付的问题；明确教学长度校验与真实入口的差异。 |
| 03 EngineCore | 重查 step、batch queue、pause/reset、输出线程。区分 Core 空闲、设备同步与客户端收到输出；补明 keep 与 clear_cache 的关系及 DP 暂停的额外协调。 |
| 04 Scheduler | 重查 running/waiting、预算裁剪、抢占、乐观计数及输出结算。此前修正继续适用；保留本章既有差额模型与例子。 |
| 05 KV Cache | 重查 get_computed_blocks、分配/释放、hash、partial 与 CoW 边界。补充 prompt_logprobs 默认跳过缓存读取及显式参数覆盖条件；增加初读自检。 |
| 06 PagedAttention | 重查 block/slot 寻址、元数据和 FlashAttention 接口。补齐完整源码路径并增加地址、请求顺序与空值的初读自检。 |
| 07 执行栈 | 重查 Worker/Runner、异步输出与状态写入。修正 staged write 图和正文，区分内容 H2D 与 UVA 元数据访问，解释轮转缓冲区和物理驻留边界。 |
| 08 编译与图 | 重查 graph 模式、并发窗口、placeholder、内存 profiling。将源码映射中的测试名前缀改为实际完整测试名，补充初读自检。 |
| 09 分布式 | 重查 TP/PP/DP/PCP/DCP 的进程数、组与通信边界。此前 PP 接收侧 gather、PCP+DP 限制等修正继续适用；补齐源码路径与基础计算自检。 |
| 10 实验 | 重查 endpoint 计时、serve 聚合与内部 IterationStats。明确 HTTP 调用前计时不等于实际网络发包时刻，内部输出时间戳也不等于 kernel 完成时间；补充实验判断自检。 |

## 关键证据

- 提交顺序：`vllm/entrypoints/offline_utils.py` 的 `_add_request`、`_run_engine`。
- 真实输入校验：`vllm/v1/engine/input_processor.py` 的 `_validate_prompt_len`、`process_inputs`；长度停止见 `vllm/v1/core/sched/utils.py` 的 `check_stop`。
- pause/reset：`vllm/v1/engine/core.py` 的 `EngineCoreProc.has_work`、`_pause_complete`、`EngineCore._finish_pause`、`_reset_caches`。
- 缓存读取条件：`vllm/sampling_params.py` 的 `SamplingParams.__post_init__` → `vllm/v1/request.py` 的 `get_skip_reading_prefix_cache` → `vllm/v1/core/kv_cache_manager.py` 的 `prefix_cache_lookup_enabled`。
- UVA：`vllm/v1/worker/gpu/buffer_utils.py` 的 `UvaBufferPool.copy_to_uva`、`StagedWriteTensor.apply_write`；构造调用见 `vllm/v1/worker/gpu/block_table.py`。
- top-k：`vllm/v1/sample/ops/topk_topp_sampler.py` 的 `apply_top_k_top_p_pytorch`、`apply_top_k_only`；penalty 计数见 `vllm/model_executor/layers/utils.py` 的 `apply_penalties`。
- 计时：`vllm/benchmarks/lib/endpoint_request_func.py` 的 completion/chat 请求函数；`vllm/v1/metrics/stats.py` 的 `IterationStats.update_from_output`。
- profiling 测试：`tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py` 的 `test_profile_cudagraph_memory_samples_and_extrapolates` 与 `test_profile_cudagraph_memory_frees_throwaway_pool`。前者主要用模拟值检查外推逻辑，后者要求 CUDA，未在本机执行。

原定义检查只是查找字符串，会让 `def test_profile_cudagraph_memory` 错误匹配到更长的
函数名。已改为用 AST 检查完整的类/函数定义，仍不把“定义存在”当作“所有运行模式均可达”。

## 已完成的本地检查

- 164 个本项目测试通过，其中第 02 章 8 个测试覆盖顺序、重复外部 ID、长度和取消边界。
- 全部 11 个 `examples/ch*.py` 教学脚本成功退出；默认演示输出与章节观察保持一致。
- 全书文档、引用及教学提示检查通过：10 章、172 个主源码锚点、588 处显式引用；章节源码映射同步维护 JSON 与 Markdown。
- 已运行一次全书 EPUB 构建与内容比对。最终发布产物在提交后再次构建，以使外部源码链接指向交付提交。

复现命令：

```bash
python3 -m unittest discover -s tests
python3 scripts/check_book.py
python3 scripts/enrich_pedagogy.py --check
python3 scripts/build_epub.py --keep-stage
python3 scripts/check_epub.py dist/vllm-source-guide.epub
```

本轮检查不把 CPU 教学程序当作生产实现，也不因静态复核通过而提升为 GPU runtime verified。
