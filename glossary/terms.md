# vLLM 源码教程术语表

本表给出全书统一使用的简明定义。章节第一次使用术语时仍需就近解释。

| 术语 | 定义 |
|---|---|
| Token | 模型处理文本时使用的离散编号单位，不一定等于一个汉字或单词。 |
| Tokenizer | 在文本与 token ID 序列之间转换的组件。 |
| Vocabulary | 模型能够识别或输出的全部 token 集合。 |
| Embedding | 将离散 token ID 映射为连续向量的查表或投影。 |
| Hidden state | token 在模型某一层中的连续向量表示。 |
| Logits | 模型对词表中每个候选 token 给出的未归一化分数。 |
| Softmax | 将一组实数分数转换为总和为 1 的概率分布。 |
| Sampling | 根据 logits、temperature、top-k、top-p 等规则选择下一个 token。 |
| Causal mask | 阻止当前位置在 attention 中读取未来 token 的约束。 |
| RoPE | Rotary Position Embedding，通过旋转 Query/Key 注入位置信息。 |
| RMSNorm | 根据向量均方根调整数值尺度的归一化方法。 |
| Attention head | 独立执行一组 Query/Key/Value attention 的子空间。 |
| GQA | Grouped-Query Attention，让多个 Query heads 共享较少的 KV heads。 |
| Autoregressive decoding | 每次基于已有 token 预测后续 token，并循环执行的生成方式。 |
| Prefill | 处理尚未计算的 prompt token 并建立 KV；可分成多个 chunk，命中前缀可跳过相应部分。 |
| Decode | 利用已有 KV Cache 逐步生成后续 token 的计算阶段。 |
| KV Cache | 保存 attention 历史 Key/Value，避免每步重复计算历史 token。 |
| Block | 普通 attention 中用于管理固定数量 token KV 的单元；状态模型可能不同，不等于 CUDA thread block。 |
| Block table | 将一个请求的逻辑 token block 映射到物理 KV block ID 的表。 |
| PagedAttention | 让 attention 通过 block table 访问非连续 KV Cache 的设计。 |
| Prefix caching | 复用不同请求间相同前缀对应的已计算 KV blocks。 |
| Continuous batching | 每个执行 step 都允许请求加入、退出或推进的动态批处理方式。 |
| Chunked prefill | 将长 prompt 的 prefill 拆成多个 token chunk，避免独占一个 batch。 |
| Preemption | 资源不足时暂停或移出请求，并在后续重新调度。 |
| Scheduler | 根据请求状态、token budget 和缓存容量决定当前 step 执行内容的组件。 |
| EngineCore | V1 中拥有 Scheduler、KV Cache 管理并协调模型执行的后端核心。 |
| Executor | 将模型执行操作派发到一个或多个 Worker 的抽象层。 |
| Worker | 管理设备环境、模型加载和执行入口的组件；是否位于独立进程由 Executor 决定。 |
| Model Runner | 准备模型输入、执行 forward、组织 sampling 和维护执行状态的组件。 |
| MRV1 / MRV2 | vLLM V1 Engine 内部的两代 Model Runner；不要与 Engine V0/V1 混淆。 |
| CUDA Graph | 捕获并重放一组 GPU 操作，以减少重复 kernel launch 的 CPU 开销。 |
| torch.compile | PyTorch 对模型图进行捕获、变换和编译的执行机制。 |
| Tensor Parallelism | 将同一层的参数和计算切分到多个设备。 |
| Pipeline Parallelism | 将不同模型层分配到不同 stage，并在 stage 间传递中间张量。 |
| Data Parallelism | 多个模型副本处理不同请求或 batch 的并行方式。 |
| Expert Parallelism | 将 MoE 的不同 experts 分布到不同设备。 |
| Collective | 多个 rank 共同参与的通信操作，例如 all-reduce、all-gather、all-to-all。 |
| TTFT | Time To First Token，从规定起点到首输出事件的时间；bench serve 从请求函数内 HTTP 调用前计时，不含客户端信号量等待。 |
| ITL | Inter-Token Latency；实际 benchmark 常按相邻输出事件采样间隔，一事件可能包含多个 token。 |
| Throughput | 单位时间内完成的请求数或处理的 token 数。 |
| Goodput | 满足指定延迟服务目标的有效吞吐量。 |

## 阅读源码所需的基础概念

这些词用于描述数组、所有权和执行顺序；先用右栏的例子理解，再回到对应章节。

| 术语 | 定义与最小例子 |
|---|---|
| Prompt | 用户提交的已有输入；文本通常先转换为 token IDs。 |
| Tensor / shape | 多维数组及各维长度；`[3,4]` 表示 3 行、每行 4 个数。 |
| Dtype | 每个数组元素的数值格式，例如 BF16；影响精度和字节数。 |
| Forward | 从输入计算输出的前向过程；推理不因此执行反向传播。 |
| MLP | 多层感知机；Transformer 中对每个 token 向量独立做非线性变换的模块。 |
| Residual | 把分支输入加回分支输出的连接；要求相加两项形状兼容。 |
| SiLU | `x/(1+exp(-x))` 形式的平滑激活函数，可用于门控。 |
| Transpose | 转置，交换矩阵的行列或张量的两个维度。 |
| Stride | 索引沿某维增加 1 时跨过的存储元素数；要换算字节还需乘元素字节数。 |
| Padding | 为统一形状补入的占位数据；应通过长度或 mask 排除其业务影响。 |
| Ragged batch | 各请求长度不同的批次；可用扁平 token 数组配合边界数组表示。 |
| Gather / scatter | 按索引收集数据 / 按索引分散写入；不等同于跨进程 all-gather。 |
| Metadata | 解释业务数据的辅助数据，例如长度、请求顺序和 block 地址表。 |
| Process / thread | 进程拥有独立地址空间；线程共享所在进程内存。 |
| Coroutine / event loop | 协程是可暂停与恢复的执行过程；事件循环在它们等待时安排其他工作。 |
| Future | 用于取得稍后结果或异常的对象；拿到 Future 不等于工作已完成。 |
| IPC / RPC | 进程间通信 / 通过通信要求另一端执行方法。 |
| ZMQ / msgpack | 本书中使用的消息传递库 / 二进制序列化格式。 |
| SSE | Server-Sent Events，服务器持续向 HTTP 客户端写入事件的格式；一条事件不一定是一 token。 |
| H2D / D2H | 主机内存到设备内存 / 设备内存到主机内存的拷贝。 |
| Pinned memory | 不被操作系统换出的主机内存，常用于支持设备异步传输；不表示 GPU 已复制完。 |
| CUDA stream / event | 设备操作队列 / 记录设备进度的同步标记；跨 stream 依赖需显式维护。 |
| Ref count | 引用计数，记录资源被持有多少次；归零后才可能再分配。 |
| Hash / cache salt | 内容指纹 / 参与缓存键的隔离信息；hash 命中不能忽略上下文语义。 |
| Free / eviction | 释放持有关系 / 删除可命中的缓存项；不是同一个操作。 |
| CoW | Copy-on-Write，写时复制；修改共享或受保护内容之前创建可写副本。 |
| FIFO / LIFO / LRU | 先进先出 / 后进先出 / 最近最少使用优先淘汰。 |
| In-flight / fence | 已提交但尚未结算的工作 / 判断相关工作何时完成的边界标记。 |
| Placeholder / stale output | 尚未知值的预留位置 / 相对抢占或重置后的状态已过期的结果。 |
| Eager / compiler pass | 直接执行模型运算 / 编译器对计算图的一次分析或改写。 |
| Dynamo / FX / Inductor | PyTorch 中捕获 Python 计算、表示计算图、生成优化代码的不同层。 |
| Guard / dynamic shape | 复用编译结果的条件检查 / 允许某些维度在规定范围内变化。 |
| Rank / world size / shard | 通信参与者编号 / 参与者总数 / 分给该参与者的数据片段。 |
| NCCL / Gloo | 常用于 GPU / CPU 的分布式通信库；一侧可用不证明另一侧可用。 |
| MoE / expert / router | 专家混合模型 / 专家子网络 / 为 token 选择专家及权重的组件。 |
| PCP / DCP | Prefill / Decode Context Parallel；PCP 增加 worker 维度，DCP 复用已有 ranks，合法组合须查配置。 |
| LSE | Log-Sum-Exp，$\log\sum_j\exp(s_j)$；保留局部 softmax 的归一化总量，供跨分块或 rank 合并。 |
| SLO / goodput | 预先规定的服务目标 / 每秒同时满足所配置目标的成功请求数。 |
| TPOT / E2EL | 首事件后平均每输出 token 时间 / 从发送到适配器规定末事件的端到端延迟。 |
| Percentile / p99 | 对样本排序后的百分位；p99 不是最大值，计算需明确插值方法与样本量。 |
| CV | 标准差除以均值，表示相对波动；均值为零时不能直接作此除法。 |
| RPS / TPS | 每秒请求数 / 每秒 token 数；TPS 必须指明输入、输出或两者总和。 |
| Benchmark / profiler | 测量性能的实验 / 解释执行时间和资源消耗的诊断工具。 |
| Pareto frontier | 没有被其他方案同时在所有目标上压过的候选集合，不自动选出唯一最佳项。 |
| Arithmetic intensity | 算术强度，每搬运一字节数据对应的浮点运算量；prefill/decode 的瓶颈还取决于 batch、模型、设备和实现。 |
| Static memory pool | 启动时按预算预分配、运行时复用的内存池；KV 池不等于全部可用显存，请求 free 通常只释放池内引用。 |
| Capture / replay | 记录操作及依赖 / 重放可执行图；只作用于已捕获且满足条件的区域，不等于整个 step 只需一次提交。 |
| Column / row parallel | 沿线性层输出维 / 输入维切分；基础 dense TP 中可配对避免中间 gather，实际通信依配置与实现而定。 |
| Little’s Law | $L=\lambda W$：在适用条件和一致观察边界下，平均系统请求数=有效到达率×平均停留时间；不单独预测 p99 或过载发散曲线。 |


## 进阶分支中会遇到的术语

这些术语用于定位可选分支。第一次阅读只需知道它们改变什么，不要求提前掌握内部算法。

| 术语 | 工作定义 |
|---|---|
| LoRA | 低秩适配，通过较小的附加参数调整模型；请求可使用不同适配器，缓存语义也需要区分。 |
| MLA | 多头潜在注意力，使用压缩的缓存表示；其容量和布局不能直接套用普通 K/V head 公式。 |
| Mamba / SSM | 状态空间序列模型及相关模型家族，维护递推状态等；统一缓存管理接口不表示里面保存的都是普通 K/V 矩阵。 |
| Speculative decoding | 推测解码：先提出草稿，再由目标模型验证，一轮可能接受多个 token，也可能拒绝并回退。 |
| EAGLE | 一类推测解码草稿方案；本书主要说明它与 Runner、调度和缓存的连接。 |
| Kernel / tile | 设备上的计算函数 / 分块处理的一小片数据；tile 不等于分配 KV 的物理 block。 |
| Triton | 编写和编译 GPU kernel 的语言与工具；看到 Python 风格源码不表示算子在 CPU 上执行。 |
| GEMM / GEMV | 矩阵乘矩阵 / 矩阵乘向量；批内多个 decode 请求仍可能组成矩阵乘法。 |
| FP16 / BF16 / FP32 | 16/16/32 位浮点格式；FP16 与 BF16 的精度和数值范围分配不同，不能只看位数判断等价。 |
| UVA | 统一虚拟寻址；本书的 UVA buffer 让 GPU 访问映射后的 pinned 主机内存，不等于把所有元数据复制进显存。 |
| TMA | 部分 NVIDIA GPU 支持的张量内存搬运机制；是否使用要看设备与 kernel 路径。 |
| DBO / microbatch | 双批次重叠 / 把批次拆为较小执行单元；能否重叠受依赖、缓冲区和后端条件约束。 |

对应缓存规格与执行边界可从 `vllm/v1/kv_cache_interface.py`、
`vllm/v1/worker/gpu/buffer_utils.py` 及第 05–09 章源码索引继续追踪。
