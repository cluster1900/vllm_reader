# 2026-09-08 全书源码与教学检查

本轮覆盖第 01–10 章正文、图示、章节源码映射、术语表、教学示例及实验说明。
面向初中级程序员和软件工程类学生，以“先直觉和小例子，再公式和源码”为阅读顺序。
采用章节通读、重点调用链静态对照、上游测试阅读、本地教学测试和 EPUB 构建检查。
以下是本轮发现并修复的问题；检查不等于证明所有平台和功能组合均无错误。

## 固定版本与边界

- Source repository: `vllm-project/vllm`
- Source path: `../vllm`
- Source commit: `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`
- Source branch / dirty: `main` / `false`，本轮只读
- Reader commit: `e41e73ec265e28074b7119aef84c024a00c0384b` 加本轮工作区修改
- Date: 2026-09-08
- Hardware: Apple Silicon，arm64
- OS / Python: macOS 26.6.2 / Python 3.14.3
- PyTorch / CUDA: 当前 Python 无 PyTorch，本机未执行 NVIDIA CUDA 路径
- vLLM install mode: 读取上游 checkout，不导入或运行上游 vLLM
- Model: 各章固定输入的标准库教学模型；未下载预训练模型权重

主核对链为：`LLM.generate` / `OpenAIServingChat` → Renderer → InputProcessor →
EngineCoreClient → EngineCore → Scheduler / KVCacheManager → Executor / Worker →
Model Runner / Attention / Sampler → Scheduler update → OutputProcessor。
另外检查 compile/graph dispatch、TP/PP/DP/EP/PCP/DCP 的关键分支及 benchmark 指标。

## 逐章修复与证据

| 章 | 发现的问题与修复 | 当前 revision 下的关键证据 |
|---|---|---|
| [01](../book/01-transformer-inference/README.md) | Prefill 表误把 layer 当作同请求并行维度；采样图让 greedy 经过温度缩放；MRV1/MRV2 顺序混写。修正分支、cached/full 对照输入和完整源码路径，补 MLP、attention、softmax 的 shape 与变量解释。 | `vllm/model_executor/models/llama.py`：`LlamaModel.forward`、`LlamaAttention.forward`；`vllm/v1/sample/sampler.py`：`Sampler.sample`；`vllm/v1/worker/gpu/sample/sampler.py`：`apply_sampling_params` |
| [02](../book/02-vllm-architecture/README.md) | 离线默认 client 与章节后文矛盾；把 SchedulerOutput 称为不可变；图遗漏单独采样。统一默认 SyncMPClient 与显式关闭 MP 的 InprocClient，补 EngineCore 派发和可选 sample。 | `vllm/envs.py`：`VLLM_ENABLE_V1_MULTIPROCESSING`；`vllm/v1/engine/llm_engine.py`：`from_engine_args`、`step`；`vllm/v1/engine/core_client.py`：`get_output`；`vllm/v1/core/sched/output.py`：`SchedulerOutput` |
| [03](../book/03-engine-core/README.md) | 队列容量简化为 2；wait pause 容易被读成排空所有旧 waiting；教学时间线未交代 token 依赖。补 PP/MRV1/MRV2 容量条件、暂停准入边界及 Future 不代表后台线程的说明。 | `vllm/config/vllm.py`：`max_concurrent_batches`；`vllm/v1/core/sched/scheduler.py`：`schedule`、`get_num_unfinished_requests`；`vllm/v1/executor/uniproc_executor.py`：`collective_rpc` |
| [04](../book/04-scheduler/README.md) | 把 computed 等同 GPU 已完成、把 stop string 写入 Scheduler；watermark 条件只按名字解释；抢占图混淆 PREEMPTED 状态与 waiting 容器；固定批次图提前结束 B。修正正文、图、实验负载与测试证据边界。 | `vllm/v1/core/sched/scheduler.py`：`schedule`、`_update_after_schedule`、`_preempt_request`；`vllm/v1/core/sched/utils.py`：`check_stop`；`tests/v1/core/test_scheduler.py`：`test_schedule_concurrent_partial_requests` |
| [05](../book/05-kv-cache/README.md) | 引用不存在的 GPUWorker 类；把池占用和物理显存增长混淆；page 公式漏补齐分支；初始化图把 specs 放到 profile 之后；watermark 与 finalized 语义过度概括。改为真实 Worker，区分物理池、引用、token 确认与设备完成。 | `vllm/v1/worker/gpu_worker.py`：`Worker`；`vllm/v1/engine/core.py`：`_initialize_kv_caches`；`vllm/v1/kv_cache_interface.py`：`AttentionSpec`；`vllm/v1/core/kv_cache_manager.py`：`allocate_slots`；`vllm/v1/core/block_pool.py`：`free_blocks`、`reset_prefix_cache` |
| [06](../book/06-paged-attention/README.md) | 主寻址例子把 null block 0 当普通数据页，与后文冲突。改为 `[2,5,3]`，同步表格、图、脚本与测试；补整除、stride、online softmax 和 LSE 的直觉说明。 | `vllm/v1/worker/gpu/block_table.py`：`BlockTables`；`vllm/v1/core/block_pool.py`：null block 初始化；`vllm/v1/attention/backends/flash_attn.py`：`do_kv_cache_update`、`forward` |
| [07](../book/07-model-execution/README.md) | 显存预算图从 free memory 推 requested；初始化次序错；完整 step 图让 Scheduler 直接调用 Executor；遗漏 connector 聚合时收多 rank 回复。修正边界与关键跳转。 | `vllm/v1/worker/utils.py`：`request_memory`；`vllm/v1/engine/core.py`：`_initialize_kv_caches`、`step`；`vllm/v1/executor/multiproc_executor.py`：`collective_rpc` |
| [08](../book/08-compile-cuda-graph/README.md) | 教学 dispatcher 错把禁用 FULL 的混合策略直接退到 NONE，原测试也固化错误；教学抢占保留 computed；stale 图无条件提前恢复。修复实现与测试，区分普通 stale 交付和 drop 分支；补缓存加载与首次编译的区别。 | `vllm/v1/cudagraph_dispatcher.py`：`dispatch`；`vllm/v1/core/sched/scheduler.py`：`_preempt_request`、waiting stale 条件；`vllm/v1/core/sched/async_scheduler.py`；`vllm/compilation/piecewise_backend.py`：构造时 compile/load 分支 |
| [09](../book/09-distributed/README.md) | 漏 PCP+DP 禁用条件且教学测试使用该组合；PP all-gather 被写在发送侧；DCP A2A 被写成 attention 前后两次交换；跨节点 PP peer 图错误；coordinator 图像逐步发令。修复条件、教学校验、通信时序及图，补 TP shape 和 LSE 数值例子。 | `vllm/config/parallel.py`：PCP/DP 校验；`vllm/distributed/parallel_state.py`：`isend_tensor_dict`、`irecv_tensor_dict`；`vllm/v1/attention/ops/dcp.py`：`dcp_a2a_lse_reduce`；`vllm/v1/engine/coordinator.py`：`_send_start_wave` |
| [10](../book/10-experiments/README.md) | E2EL 的适配器差异未说明；TTFT 排障图混入客户端发送前排队；token/s 除 RPS 被称为每用户速率；Prometheus counter 名称和比例单位不清；教学分析器假定每事件一 token。修正文档与分析器，支持实际 completion token 数及末响应时间。 | `vllm/benchmarks/lib/endpoint_request_func.py`：completion/chat adapters；`vllm/benchmarks/serve.py`：`calculate_metrics`、`get_request`；`vllm/benchmarks/sweep/plot_pareto.py`：`_infer_user_count`；`vllm/v1/metrics/loggers.py` |

上述静态证据说明固定 revision 的代码行为；阅读上游测试不等于已经运行上游测试。
例如 watermark 的注释说“已调度请求”，但 Scheduler 实际传入 `bool(self.running)`，
本轮以调用点为准；不能仅复述参数名或注释。

## 教学与维护修复

- 十章明确标注初中级程序员与软件工程学生受众；每章提供具体第一遍小节路线、术语入口、一道带解释的小练习。
- 保留原有深入内容与题库，长题库前 5 题作为首轮检查，其余留作第二遍。
- 补张量 shape、公式变量、时间与字节单位，以及 Future、IPC、rank、CoW、LSE、p99 等术语。
- 全书目录补回第 06 章的必要寻址前置，章节映射同步使用真实 `Worker` 类名。
- 教学提示脚本改为只补缺口，保留已人工编辑的阅读提示，避免运行构建时覆盖源码边界解释；有回归测试。
- 15 张过宽流程图改为纵向或明确子图方向，关键图的调用/返回边同步复核。
- 历史实验记录保留日期；本轮 CPU 复跑单独记录，GPU 空白模板不伪装成实测。

## 本轮验证

已经完成以下检查，结果均通过：

| 检查 | 本轮结果 |
|---|---|
| `python3 scripts/check_book.py` | 10 章、172 个主源码锚点、488 处引用检查通过 |
| `python3 scripts/enrich_pedagogy.py --check` | 0 章需要补写；已编辑提示保持不变 |
| `python3 -m unittest discover -s tests` | 138 个测试通过，含本轮新增的 8 个边界/回归测试 |
| 所有 `examples/ch*.py` | 11 个脚本退出码均为 0 |
| `python3 scripts/build_epub.py --keep-stage` | 完整构建通过；之后对视觉复核修改的 17 张图重新渲染，并从当前全部正文重新打包 |
| `python3 scripts/check_epub.py dist/vllm-source-guide.epub` | 15 个 XHTML、297 张图片、81 个 MathML 公式、294 份读图说明通过结构检查 |
| 当前正文与最终构建暂存正文比较 | 10 章逐一一致 |
| `git diff --check` | 通过 |
| 上游工作区复查 | 仍为 `main`，无未提交修改 |

逐章脚本的完整复跑命令为：

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
```

关键实际观察：第 01 章 cached/full logits 最大差为 0；第 04 章仍输出四轮
`A:p6` → `A:p6,B:p2` → `A:d1,B:p1,C:p2` → `A:d1,B:d1,C:d1`；
第 06 章 `[2,5,3]` 映射得到 `[8,9,10,11,20,21,22,23,12,13]`；第 09 章
TP=2、PP=2、DP=2、PCP=1 仍得到 8 个 Worker。它们只验证教学模型与正文契约。

视觉复核共修改 41 张图，其中 15 张过宽流程图调整为纵向或明确子图方向。
全书渲染后不存在“宽度大于 1600 且宽高比超过 7”的超宽图；抽查采样、调度、完整
执行、TP/PP 和客户端计时图，文字及箭头可辨认。该尺寸检查和抽查不等于逐页验证
所有 EPUB 阅读器；最终产物为 `dist/vllm-source-guide.epub`（按仓库构建命令生成）。

构建出现 Pandoc 的 `zh-CN` 内置翻译提示，未阻止输出；已检查目录与各章一级标题为
中文，章节、图片和 MathML 结构完整。

## 保留的验证边界

本轮没有运行 vLLM GPU 单元测试、真实模型生成、CUDA Graph capture/replay、NCCL/Ray
多卡通信或性能 benchmark。章节保持 `draft`、`runtime_verified=false`。更新的
`verified_at` 记录本轮静态核对日期，不表示已完成设备运行验证。

下一步可在固定 revision 的 NVIDIA 环境，按各章实验协议执行：单 GPU 请求/取消 trace、
Scheduler/KV 压力、不同 Runner 的数值对照、graph 回退、多 rank 通信与 benchmark
适配器指标对照，再保存实际模型、配置、命令、输出和异常。不存在 GPU 实测证据的
性能数字仍只能作为教学例子。
