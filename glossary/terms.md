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
| Prefill | 首次并行处理 prompt token，生成初始 KV Cache 的计算阶段。 |
| Decode | 利用已有 KV Cache 逐步生成后续 token 的计算阶段。 |
| KV Cache | 保存 attention 历史 Key/Value，避免每步重复计算历史 token。 |
| Block | vLLM 管理一组固定数量 token 的 KV 存储单元，不等于 CUDA thread block。 |
| Block table | 将一个请求的逻辑 token block 映射到物理 KV block ID 的表。 |
| PagedAttention | 让 attention 通过 block table 访问非连续 KV Cache 的设计。 |
| Prefix caching | 复用不同请求间相同前缀对应的已计算 KV blocks。 |
| Continuous batching | 每个执行 step 都允许请求加入、退出或推进的动态批处理方式。 |
| Chunked prefill | 将长 prompt 的 prefill 拆成多个 token chunk，避免独占一个 batch。 |
| Preemption | 资源不足时暂停或移出请求，并在后续重新调度。 |
| Scheduler | 根据请求状态、token budget 和缓存容量决定当前 step 执行内容的组件。 |
| EngineCore | V1 中拥有 Scheduler、KV Cache 管理并协调模型执行的后端核心。 |
| Executor | 将模型执行操作派发到一个或多个 Worker 的抽象层。 |
| Worker | 管理一个 accelerator 进程、设备环境、模型加载和执行入口的组件。 |
| Model Runner | 准备模型输入、执行 forward、组织 sampling 和维护执行状态的组件。 |
| MRV1 / MRV2 | vLLM V1 Engine 内部的两代 Model Runner；不要与 Engine V0/V1 混淆。 |
| CUDA Graph | 捕获并重放一组 GPU 操作，以减少重复 kernel launch 的 CPU 开销。 |
| torch.compile | PyTorch 对模型图进行捕获、变换和编译的执行机制。 |
| Tensor Parallelism | 将同一层的参数和计算切分到多个设备。 |
| Pipeline Parallelism | 将不同模型层分配到不同 stage，并在 stage 间传递中间张量。 |
| Data Parallelism | 多个模型副本处理不同请求或 batch 的并行方式。 |
| Expert Parallelism | 将 MoE 的不同 experts 分布到不同设备。 |
| Collective | 多个 rank 共同参与的通信操作，例如 all-reduce、all-gather、all-to-all。 |
| TTFT | Time To First Token，请求到达后得到首个输出 token 的时间。 |
| ITL | Inter-Token Latency，连续输出 token 之间的延迟。 |
| Throughput | 单位时间内完成的请求数或处理的 token 数。 |
| Goodput | 满足指定延迟服务目标的有效吞吐量。 |
