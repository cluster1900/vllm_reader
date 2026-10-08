# 全项目文档复核与修订记录

本轮覆盖仓库首页、十章正文、阅读指南、术语表、十份实验文档、源码映射、历史审阅记录、
网页说明与 EPUB。检查分为全库结构与引用检查、重点调用链和边界的静态源码复核、CPU
教学运行、导出一致性与显示检查。结构检查通过不表示所有运行模式和 kernel 都已验证。

## 版本与环境

- Source repository / commit：`vllm-project/vllm` / `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`。
- Source path / branch / dirty：`../vllm` / `main` / `false`；上游保持只读。
- Reader base：`333377732ab913096119440433a890286409959b`，加本报告所在提交的文档修订；以下验证在提交前完成。
- Date：2026-10-08。
- Hardware / OS / Python：Apple M2，arm64，macOS 26.6.2，Python 3.14.3。
- PyTorch / CUDA / vLLM install mode：当前解释器没有安装 PyTorch、vLLM；未运行 CUDA，
  vLLM 仅用于静态源码阅读。
- Model：随书标准库教学模型，未下载或运行预训练权重。
- 验证强度：章节保持 `draft`、`runtime_verified: false`；`verified_at` 是静态核对日期。

## 源码地图与复核问题

复核沿既有主线展开：`LLM.generate` / `OpenAIServingChat` → Renderer → InputProcessor →
EngineCoreClient → EngineCore → Scheduler / KVCacheManager → Executor / Worker → Model Runner →
Sampler → Scheduler 输出结算 → OutputProcessor。重点检查以下问题：

1. 默认值、可选分支与“所有模式都如此”的表述是否相符。
2. 请求数据、缓存引用、物理显存和异步结果是否被混为同一种状态。
3. 正文、图示、术语、实验命令与实际指标口径是否一致。

章节级锚点同步维护在 [JSON 映射](chapter-source-map.json) 与
[可读映射](chapter-source-map.md)。

## 逐章修正

| 章 | 问题与修正 | 固定 revision 下的证据 |
|---|---|---|
| 01 | 普通 K/V 容量公式容易被直接套到 MLA；明确单份潜在缓存表示与 spec 边界，重写容量图和阅读路线的图注。 | `vllm/v1/kv_cache_interface.py`：`AttentionSpec`、`MLAAttentionSpec`；`vllm/model_executor/layers/attention/attention.py`：`Attention.__init__` |
| 02 | 输出容器被限定为 ZMQ 类型，且误称包含该 client 所有请求；改为同进程也使用、按需交付的容器，补可选 utility 字段。 | `vllm/v1/engine/__init__.py`：`EngineCoreOutputs`；`vllm/v1/engine/core_client.py`：`InprocClient.get_output` |
| 03 | 对进程隔离与 GIL 的工程解释被写成已证实的设计收益；明确标为推断，复查 step、batch queue 与取消顺序。 | `vllm/v1/engine/core.py`：`EngineCore.step`、`step_with_batch_queue`、`EngineCoreProc.run_busy_loop` |
| 04 | 抢占被写成丢弃所有状态并从 prompt 重算；改为保留已接受 token、释放缓存引用、重置进度，说明重新命中和延迟释放；目标计数改用真实属性表达式。 | `vllm/v1/core/sched/scheduler.py`：`_preempt_request`、`_free_request_blocks`、`schedule`；`vllm/v1/request.py`：`num_tokens_with_spec` |
| 05 | “未来才支持局部尾块”与本章现有 partial-tail 路径矛盾；限定完整 block 主线，区分注释的设计意图与当前条件分支。 | `vllm/v1/core/kv_cache_manager.py`：`get_computed_blocks`；`tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py`：`test_hybrid_mamba_partial_tail_owner_uses_cow_on_continue` |
| 06 | MRV2 的 UVA 说明没有区分写入内容和元数据；补齐 H2D、UVA、apply-write、gather 的边界，以及多 group 合并写入。 | `vllm/v1/worker/gpu/buffer_utils.py`：`StagedWriteTensor.apply_write`、`FusedStagedWriter.apply`；`vllm/v1/worker/gpu/block_table.py`：`apply_staged_writes`、`gather_block_tables` |
| 07 | Graph 显存估算默认开关写反；greedy/random 被描述为择优；改为默认 `1`、按请求 temperature 选择，并区分 logits 与 logprobs。 | `vllm/envs.py`：`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`；`vllm/v1/worker/gpu_worker.py`：`determine_available_memory`；`vllm/v1/sample/sampler.py`：`forward`、`sample` |
| 08 | CUDA Graph 类比被叫作硬件指令流；batch queue 叙述漏 grammar 依赖例外；改为操作及依赖的编排，补图中延迟采样和显存预算变量、适用范围。 | `vllm/v1/engine/core.py`：`step_with_batch_queue`；`vllm/v1/worker/gpu_worker.py`：`determine_available_memory` |
| 09 | 单一回复 rank 被解释为只有 TP rank 0 聚合；通信字节减少被直接写成大幅收益。区分输出路由与分布式采样，说明额外通信成本。 | `vllm/v1/executor/multiproc_executor.py`：`_get_output_rank`、`collective_rpc`；`vllm/v1/worker/gpu/model_runner.py`：`sample`；`vllm/distributed/parallel_state.py`：`isend_tensor_dict`、`irecv_tensor_dict` |
| 10 | Pareto 流程图与坐标图说明不符，支配条件遗漏“某项持平”；重做可手算的示例，区分上游双吞吐坐标和保留持平点的规则；统一 HTTP 调用前的计时起点和 block 微基准含义。 | `examples/ch10_benchmark_reasoning.py`：`pareto_frontier`（本项目）；`vllm/benchmarks/sweep/plot_pareto.py`：`_pareto_frontier`、`_prepare_records`；`vllm/benchmarks/lib/endpoint_request_func.py`；`benchmarks/benchmark_block_pool.py` |

这些证据包括实际源码阅读与测试用例阅读，不能表述为已运行上游 GPU 测试。
例如 Worker 中仍有“opt-in”的注释，但 `envs.py` 的实际默认值为 `1`，本轮以执行代码为准。

## 其他文档与格式

- 术语表同步抢占、Pareto、TPOT/E2EL；补足图注中的箭头含义与非真实枚举节点说明。
- 修复输出类型段落的列表间距，统一证据标签后的空格；保留原有章节结构和有效内容。
- 网页将教学程序数修正为 11；纠正输入预处理与 client 的关系，补充返回路径和验证边界；
  首页大标题按完整词组换行，避免将“一条”“调用链”拆开。
- 第 05、06 章实验命令不再依赖作者电脑绝对路径；第 07 章给出使用本地模型路径的 Runner
  延迟对照命令，并说明它不导出 token 正确性结果。
- 实验文件增加本轮结果入口；GPU 模板补 Reader revision、计时起点或 graph 开关等关键字段。
  历史输出保留原日期，不把旧记录改写成本轮运行结果。
- 首页补全 Pandoc、Node.js、Python 与浏览器构建依赖；历史审阅报告标明其数量不代表当前状态。
- EPUB 版本日期更新为本轮日期，正文仍从 Markdown 构建。
- EPUB 显示检查发现长源码路径、表格和高亮代码在窄屏溢出；样式增加路径换行，并覆盖
  Pandoc 高亮代码的 `white-space: pre` 规则。修改只影响显示，不改变可复制的源码文本。

## CPU 实验记录

预期：11 个脚本退出码均为 0，已有测试通过，关键输出与各章教学例子一致。
在仓库根目录复现，脚本不依赖 vLLM 或 GPU：

```bash
python3 examples/ch01_minimal_inference.py
python3 examples/ch01_numerical_walkthrough.py
python3 examples/ch02_request_lifecycle.py
python3 examples/ch03_engine_core_loop.py
python3 examples/ch04_continuous_batching.py
python3 examples/ch05_kv_cache_blocks.py
python3 examples/ch06_paged_addressing.py
python3 examples/ch07_execution_pipeline.py
python3 examples/ch08_async_compile_graph.py
python3 examples/ch09_parallel_topology.py
python3 examples/ch10_benchmark_reasoning.py
python3 -m unittest discover -s tests
```

实际观察：11 个脚本全部成功退出，关键结果如下。耗时相关数字是教学模型的固定输入或计算值，
不表示当前电脑的性能测量。

| 章 | 本轮观察 |
|---|---|
| 01 | 3 轮 cached/full logits 最大差均为 0；QK score 计数为 333136 与 12720；数值 attention 输出为 `[1.2309, 0.9047]`。 |
| 02 | 两个离线输出保持提交顺序；在线三个增量输出后能交付 abort。 |
| 03 | 队列先提交，再逐轮结算；最后一次排空并得到 length 终态。 |
| 04 | 四轮工作为 `A:p6` → `A:p6,B:p2` → `A:d1,B:p1,C:p2` → `A:d1,B:d1,C:d1`；最终 KV 计数为 0。 |
| 05 | B 命中 6 token；继续写触发 block 2 到 block 3 的 CoW。 |
| 06 | `[2,5,3]` 映射到 `[8,9,10,11,20,21,22,23,12,13]`。 |
| 07 | 返回请求 A、B 与二维 sampled token 列表 `[[4],[5]]`。 |
| 08 | 理想流水模型为 28 ms / 22 ms；uniform/non-uniform 分别选择 FULL / PIECEWISE。 |
| 09 | 每 DP engine 4 个 Worker，跨 DP 共 8 个；TP/PP/DP 分组与教学拓扑一致。 |
| 10 | 固定样本完成 2 请求；10 req/s、25 token/s、p99 TTFT 119.8 ms。 |

结论：本轮教学程序输出与文档契约相符。限制：程序使用简化状态与固定数据，不验证真实
模型质量、ZMQ 时序、设备执行、NCCL 或生产性能。

## 文档与导出验证

复核命令：

```bash
python3 scripts/check_book.py
python3 scripts/enrich_pedagogy.py --check
python3 scripts/build_epub.py --keep-stage
python3 scripts/check_epub.py dist/vllm-source-guide.epub
git diff --check
```

最终结果：

| 检查 | 结果 |
|---|---|
| `python3 -m unittest discover -s tests` | 164 项测试通过。 |
| 全部 `examples/ch*.py` | 11 个脚本退出码均为 0。 |
| `python3 scripts/check_book.py` | 10 章、176 个主源码锚点、626 处引用检查通过；同时检查全库 Markdown 链接、围栏和空白。 |
| `python3 scripts/enrich_pedagogy.py --check` | 0 章需要补写，已有人工说明保持不变。 |
| Pandoc 解析与额外符号核对 | 33 份原有 Markdown 均可解析；22 处 `路径::符号` 引用的定义存在。新增本报告另行解析通过。 |
| 全部 Mermaid 渲染 | 294 张图成功渲染；最终打包复用前逐张核对嵌入的 Mermaid 源文完全一致。 |
| `python3 scripts/check_epub.py dist/vllm-source-guide.epub` | 12 份正文来源、58,182 条有序语义记录一致；15 个 XHTML、462 张打包图片、174 处 SVG 公式，封面与锚点通过。 |
| EPUB 显示检查 | 从最终 EPUB 解包，在 Chromium 的 820/390 像素宽度各检查 14 个阅读页面；无页面横向溢出、无损坏图片。另目视抽查 Pareto、采样、容量公式、源码表格与代码。 |
| 公式预览 | `python3 scripts/build_math_preview.py` 生成 174 处公式、22 页；1280/390 像素宽度共 44 次页面检查，无页面溢出或损坏图片，目视抽查行内与块公式。 |
| 网页 | 本地静态页面在 1440/390 像素宽度无页面溢出、图片损坏、缺失锚点或脚本错误；章节外链的仓库路径存在。未发布到 GitHub Pages。 |
| `git diff --check` 与上游状态 | 通过；上游仍为锁定 commit、`main`、clean。 |

首次 EPUB 打包时，第 06 章在渲染期间又补充了多 group 写入边界，一致性检查准确发现正文
不同步。随后从当前全部 Markdown 重新生成正文、公式和容器，复用源文一致的已渲染图，
最终完整校验通过；交付文件不含这次中间状态。上面的常规构建命令可以从最终工作区重现结果。

Pandoc 仍提示缺少 `zh-CN` 的内置翻译和 Abstract 词条；中文正文、目录和结构检查未受阻。
Chromium 中的显示结果不等于所有第三方 EPUB 阅读器、字体和分页模式都已验证。

## 保留边界

未执行真实 NVIDIA 模型生成、CUDA Graph capture/replay、NCCL/Ray、多机通信和性能 benchmark。
后续应在固定 revision 的 GPU 环境按各章实验协议记录模型、硬件、最终配置、命令与实际输出。
以上网页检查在本地完成，未验证 GitHub Pages 的部署结果。EPUB 的仓库外链按构建时的
Reader HEAD 固定，正式交付时应在提交后重新构建，使外链与本轮 Markdown 修订一致。
