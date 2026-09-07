# 章节源码映射

本文件回答“十章是否与当前代码一一对应”。映射基线为
`5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`，机器可读版本见
`meta/chapter-source-map.json`。

这里的“一一对应”不是指一章只能对应一个文件，而是指：每章有清楚且互不混淆的
核心问题、主实现入口和扩展边界；相邻章节可以共享调用链，但不能重复承担同一层的
完整解释责任。

| 章 | 教学责任 | 当前源码主锚点 | 明确边界 |
|---|---|---|---|
| 01 | 建立自回归推理、prefill/decode、KV 与 sampling 基础 | `llama.py`、`attention.py`、`sampling_params.py`、MRV1/MRV2 sampler | 只建立模型计算直觉，不解释服务调度 |
| 02 | 给出离线/在线请求的端到端地图 | `entrypoints/llm.py`、offline utils、Renderer、OpenAI chat serving、`AsyncLLM`、core client、输入/输出处理器 | ZMQ 多进程是在线主线；单 rank `UniProcExecutor` 的 Worker 位于 EngineCore 进程，不能把逻辑 Worker 一律画成额外进程 |
| 03 | 解释 Engine 门面、client、EngineCore 生命周期和 IPC | `LLMEngine`、`EngineCoreClient`、`EngineCore`、`EngineCoreProc`、`EngineCoreActor` | 不展开 Scheduler 算法和 GPU tensor 构造 |
| 04 | 解释 token 级调度与 Continuous Batching | `Scheduler`、`SchedulerOutput`、`RequestQueue`、`Request` | 负责“算谁、算多少”，不负责 kernel 如何计算 |
| 05 | 解释 KV 规格、启动容量和 block 生命周期 | `KVCacheSpec`、`get_kv_cache_configs`、`GPUWorker.determine_available_memory`、`KVCacheManager`、`BlockPool`、coordinator、single-type managers | 常规命中以完整 block 为主；当前混合 KV 路径还支持条件化 partial-tail 命中和 CoW；不展开 kernel 寻址 |
| 06 | 解释 block table、slot mapping、ragged metadata 到 attention backend/kernel 的寻址与执行契约 | `Attention` 与 unified ops、backend interface/registry/selector、CUDA platform、MRV1/MRV2 block table、FlashAttention、特定 `PagedAttention` helper | 历史 PagedAttention kernel 与当前实际 backend 必须分开；FlashAttention 等实现也可直接消费分页 KV |
| 07 | 解释 SchedulerOutput 如何成为 GPU forward 和采样结果 | UniProc/Multiproc Executor、WorkerWrapper/GPU Worker、model registry/loader、MRV1/MRV2 runner、sampler 与 outputs | unique reply 不等于单 rank 执行；V1 Engine 与 MRV1/MRV2 是两个维度；MRV2 是受条件约束的默认路径 |
| 08 | 解释异步调度、compile 和 CUDA Graph 的开销优化 | `AsyncScheduler`、EngineCore batch queue、compile decorator/backend/ranges/cache、MRV1 dispatcher/wrapper、MRV2 graph manager | batch queue 与 async scheduler 分开；配置模式与 runtime mode 分开；`torch.compile` 与 CUDA Graph 分层讲；MRV2 manager 不是旧 wrapper 的别名 |
| 09 | 解释 TP/PP/DP/EP/PCP/DCP 拓扑、group、ownership 和通信恢复语义 | `ParallelConfig`、`parallel_state.py`、TP linear、PP model path、DP client/coordinator、MoE runner、PCP manager、DCP ops、MP/Ray executors | 当前 worker `world_size=TP×PP×PCP`，DCP 不扩进程；EP 展平 DP×PCP×TP；Ray V1/V2 由 factory 选择 |
| 10 | 建立可复现的 benchmark、metrics、profiling、归因与回归方法 | latency/throughput/serve benchmark、endpoint timestamps、datasets、sweep/Pareto、V1 metrics、ProfilerConfig、组件微基准 | 客户端与内部计时边界分开；性能结论必须绑定 workload、噪声、配置和正确性验证 |

## 审计修正

本轮对原始十章设想做了以下结构性修正：

1. 将单独的 Continuous Batch 章并入 Scheduler，因为当前 V1 Scheduler 以统一的
   token budget 在每个 step 同时推进 prefill/decode，请求动态进出正是调度主循环。
2. 新增 vLLM 全景章，否则不熟悉 LLM 的读者会在 Engine、Scheduler、Worker 和
   Attention 之间失去端到端位置感。
3. 将 Worker 扩展为 Executor、Worker、Model Runner 与 Sampler 的完整执行栈，匹配
   当前代码的真实分层。
4. 将 CUDA Graph 章扩展为异步调度、`torch.compile` 与 CUDA Graph，避免把三种不同
   优化机制混成一个概念。
5. 将 Multi GPU 扩展为 TP/PP/DP/EP/PCP/DCP 和 executor backend，因为当前
   `ParallelConfig` 已经不只是传统 TP。
6. 修正“prefix cache 只复用完整 block”的绝对表述：常规契约仍以完整 block 为主，
   但当前混合 KV 实现已有细粒度 partial-tail entry、命中和 copy-on-write。
7. 修正“PagedAttention 是单一当前 kernel/统一类”的表述：分页 KV 是跨 backend 的地址
   契约；当前 FlashAttention 直接接收 block table，而 `ops/paged_attn.py` helper 的当前
   Python 调用集中在 ROCm attention 路径。
8. 修正“Worker 直接等于模型执行”和“V1 Engine 必然使用 MRV1”的表述：Executor 向所有
   rank 广播执行命令但通常只读取一个 output rank 的回复；GPU Worker 在分布式初始化和显存
   快照后构造 MRV1/MRV2，model registry 与 checkpoint loader 还是两个独立选择维度。
9. 修正“异步调度、batch queue 和异步 D2H 是同一机制”的表述：PP 也会启用 batch queue；
   `AsyncScheduler` 通过 output placeholder 表示未确认进度；MRV2 `AsyncOutput` 则在独立 copy
   stream 上搬运结果，三者的状态与同步边界不同。
10. 修正“FULL_AND_PIECEWISE 是运行时 graph mode”的表述：它与 FULL_DECODE_ONLY 是复合配置；
    `ForwardContext` 中合法 runtime mode 只有 NONE、PIECEWISE 和 FULL，最终模式还会受 attention
    backend、shape、LoRA、cascade、encoder、DP/DBO 等条件影响。
11. 修正“worker world size 永远等于 TP×PP”的表述：当前 `ParallelConfig` 的实际赋值包含
    PCP，即 TP×PP×PCP；跨 DP 总 worker 数再乘 DP，DCP 则复用已有 ranks，不增加进程。
12. 修正“MoE EP 仍把单个 expert 按 TP 切分”的表述：当前 EP 模式把 DP×PCP×TP 展平为
    EP size，让每个设备拥有完整 experts；route 后通过 dispatch、local compute、combine 恢复
    token 语义。
13. 修正 expert placement 注释中的余数归属：`determine_expert_map` 的实际代码把不能整除的
    experts 分给前面的 ranks，而不是 docstring 所称的最后 ranks。
14. 明确 PCP 与 DCP 不是同一功能：PCP 增加 worker 并切 prefill chunks；DCP 复用 ranks 切
    decode context，partial attention 必须用 LSE 权重合并，不能直接平均。
15. 修正“TTFT 包含客户端并发队列”的表述：当前 `bench serve` 在获得 semaphore 后才记录
    request `start_time`，TTFT/E2EL 从真实 HTTP send 计算；semaphore 等待单独写入
    `client_queue_time`，但仍影响整个 benchmark duration 与完成吞吐。
16. 修正“ITL 与 TPOT 是同一指标”的表述：ITL 是流式事件间隔样本，长请求贡献更多样本；
    TPOT 是每请求的 `(E2EL-TTFT)/(output_tokens-1)`，二者权重和 backend chunk 语义不同。
17. 修正“日志 prompt throughput 等于业务输入吞吐”的表述：`LoggingStatLogger` 只统计本地
    computed prompt tokens，排除 local prefix hit 和 external KV transfer；serve benchmark 的
    input token throughput 则按成功请求的业务输入计数。
18. 补充 sweep 工具的资源计数边界：当前 Pareto GPU 推导使用 TP×PP×DP，未包含 PCP；含
    PCP 的实验应显式提供 GPU count，不能直接套用默认推导。

## 复查命令

```bash
python3 scripts/check_book.py
```

校验脚本会检查上游 HEAD、十章 frontmatter、映射文件中的源码路径和关键符号。它能
发现结构漂移，但无法证明章节中的控制流解释和性能结论正确；这些仍需测试和实验。
