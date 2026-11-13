# 符号与记法约定

本文件回答一个问题：书里的每个字母到底指什么，换一章会不会变。全书约 400 个公式（行间与行内）原先
各自在附近定义符号，同一个字母在不同章里常常是不同的量（例如 `T` 先后表示 token 数、温度、block
大小和耗时），这里统一成一张表，并给出旧写法到新写法的对照。

使用方法：

- 读正文时遇到不确定的字母，先查第 1 节的全局符号表；表里没有的，是该小节自己定义的局部字母（第 3 节）。
- 写或改公式时，按第 2、5 节的规则选符号，按第 6 节的模板写讲解，按第 7 节决定要不要配图。
- 第 4 节是改名对照表：各章正文已经按它改成新写法；表里的旧写法只用来识别旧资料和遗留写法。
  表里按符号查旧写法，不按行号。

源码里的名字（如 `num_computed_tokens`）不受本约定约束，始终写成代码样式；公式里用的是它们的数学记号，
两者的对应关系在表里列出。

## 1 全局符号表

“全局”的意思是：在全书任何位置，这个字母（带这个下标）只表示表里的那一个量。
没有写单位的量是无单位的比例、下标或标志位。

### 1.1 形状与规模

| 符号 | 含义 | 单位 / shape | 备注 |
|---|---|---|---|
| $B$ | 批大小：一批里的请求（序列）数，只出现在教学形状 `[B,S,…]` 里 | 个 | block 大小另写 $B_{\mathrm{blk}}$，不再用 $B$ |
| $S$、$S_r$ | 单个请求的序列长度：该请求已有、需要存 KV 的 token 数；$S_r$ 是请求 $r$ 的 | token | ch01、ch05、ch06；与 ch04 的 $N_{\mathrm{tokens}}$ 是同一类量的两个视角（KV Cache 一侧、scheduler 一侧），各章保留原名 |
| $T$ | 一个 step 展平后的 token 总数，即 vLLM 形状 `[T,…]` 的第 0 维，$T=\sum_r N_{q,r}$ | token | 不再表示温度、block 大小或耗时 |
| $D$、$D_{\mathrm{in}}$、$D_{\mathrm{out}}$、$D_{\mathrm{ff}}$ | hidden size；$D_{\mathrm{in}}$、$D_{\mathrm{out}}$ 是某个线性层的输入、输出宽度；$D_{\mathrm{ff}}$ 是 MLP 中间宽度 | 元素 | 旧 `I`（MLP 宽度）见第 4 节 |
| $H_q$、$H_{\mathrm{kv}}$、$H_{\mathrm{kv,local}}$、$H_k$、$H_v$ | Query head 数、KV head 数；$H_{\mathrm{kv,local}}=H_{\mathrm{kv}}/\mathrm{TP}$ 是一张卡上的 KV head 数 | 个 | $H_k$、$H_v$ 只在 K、V head 数可能不同的通式里出现 |
| $d_h$、$d_k$、$d_v$ | 单个 head 的宽度（Query/Key，通常 Value 相同）；缩放因子写 $\sqrt{d_h}$ | 元素 | $D=H_q d_h$ 只在各 head 等宽时成立；$d_k$、$d_v$ 只在 K、V 宽度可能不同的通式里出现 |
| $L$ | Transformer 层数 | 层 | 不再表示长度、lookahead 或 LSE |
| $\lvert\mathcal V\rvert$ | 词表大小；写在张量形状里时用 `vocab_size` | 个 | $V$ 只表示 Value 矩阵 |
| $B_{\mathrm{blk}}$ | block 大小：一个 KV block 容纳的 token 数，代码名 `block_size` | token/block | 旧 $T$（ch05）、$B$（ch06） |
| $N_q$、$N_k$ | ch01 单头例子里 query、key 的行数（`Q:[N_q,d_h]`、`K:[N_k,d_h]`） | 行 | 同族的 $N_{q,r}$ 是请求 $r$ 本步的 query token 数 |

### 1.2 模型、概率与采样

| 符号 | 含义 | 单位 / shape | 备注 |
|---|---|---|---|
| $X$ | 一层的输入（hidden state） | `[T,D]` | |
| $Q$、$K$、$V$ | Query、Key、Value 矩阵；小写 $q$、$k$、$v$ 是其中一行（向量） | `[T,H,d_h]` 或单头 `[N,d_h]` | $V$ 不表示词表；$Q$ 不表示吞吐，吞吐写 $\mathrm{thr}$（第 1.6 节） |
| $W_Q$、$W_K$、$W_V$、$W_{\mathrm{gate}}$、$W_{\mathrm{up}}$、$W_{\mathrm{down}}$ | 投影与 MLP 的权重矩阵，一律带下标 | | 世界规模写 $W_{\mathrm{engine}}$ 等，两者靠下标区分；ch09 的通用线性层权重写 $A$（局部，见第 3 节） |
| $O$ | attention 输出；$O_i$ 是 KV 分片 $i$（ch09 的 DCP shard）给出的局部输出，$i$ 是分片序号 | `[T,H_q·d_v]` | 不再表示输出 token 数 |
| $\mathrm{Mask}$ | causal mask（可见位置 0，不可见位置 $-\infty$） | `[N_q,N_k]` | 旧 $M$，把 $M$ 留给显存 |
| $s_j$、$a_j$ | 第 $j$ 个 key 的打分 $q\cdot k_j/\sqrt{d_h}$；softmax 之后的 attention 权重 $a_j$ | 无单位 | 旧 ch06 的 $p_j$；向量形式 $a$ 见 ch01 |
| $z_i$、$p_i$ | 候选 token $i$ 的 logit 与概率（softmax 之后，总和为 1） | 无单位 | $p_i$ 只表示概率 |
| $\tau$ | softmax 温度，$\tau>0$ | 无单位 | 旧 $T$（ch01、ch07） |
| $\mathrm{top\_k}_i$、$\mathrm{top\_p}_i$ | 第 $i$ 行请求的 top-k、top-p 参数；不分行时写 $\mathrm{top\_k}$、$\mathrm{top\_p}$ | 个 / 概率 | 旧 $k_i$、$p_i$（ch01） |
| $\operatorname{LSE}$、$\operatorname{LSE}_i$ | Log-Sum-Exp：一个 query 对一组 key 的 $\log\sum_j\exp(s_j)$；$\operatorname{LSE}_i$ 是 KV 分片 $i$ 上的值（求和只覆盖该分片的 key，$i$ 是分片序号，不是 query 行号） | 无单位 | 旧 ch09 的 $L$、$L_i$；ch06、ch09 都按这个含义使用 |
| $\mathcal H(x)$ | top-$k$ 熵（ch07 §7.22.1 的 FLy 判定） | 无单位 | 旧 $H(x)$ |
| $\theta$ | RoPE 的旋转角 | 弧度 | ch01 的 RoPE 公式；全书只有这一个 $\theta$，因此不再列入第 3 节的局部字母 |
| $\eta_i$ | 第 $i$ 个 block 的链式 hash，$\eta_i=\operatorname{hash}(\eta_{i-1},\mathrm{tokens}_i,\dots)$；图里写 `hash_i` | | 旧 $H_i$（ch05） |
| $(\cdot)^{\top}$ | 转置 | | 旧 `\mathsf{T}` 与 `K^T` |

### 1.3 token 与其他计数：$N_{\dots}$ 与 $n_{\dots}$

规则：**$N_{\dots}$ 的单位是 token**（包括 ch10 的输入、输出 token 总数 $N_{\mathrm{in}}^{\Sigma}$、$N_{\mathrm{out}}^{\Sigma}$）；
**$n_{\dots}$ 是其他计数**：block 数、chunk 数、请求数、用户数、GPU 数、通信阶段数。ch10 里的请求数、用户数、GPU 数因此写 $n_{\mathrm{req}}$、
$n_{\mathrm{users}}$、$n_{\mathrm{GPU}}$，不写 $N_{\dots}$。

| 符号 | 含义 | 出处 |
|---|---|---|
| $N_{\mathrm{prompt}}$、$N_{\mathrm{out}}$、$N_{\mathrm{spec}}$ | prompt token 数；已确认的输出 token 数；本步的 speculative token 数 | ch01 §计算量、ch04、ch10（$N_{\mathrm{out}}$：单个请求的输出 token 数，TPOT 的分母） |
| $N_{\mathrm{target}}$、$N_{\mathrm{tokens}}$ | 请求这一步要算到的目标 token 数；请求当前已知的 token 总数（`num_tokens`） | ch04 |
| $N_{\mathrm{computed}}$、$N_{\mathrm{new}}$ | 已计算的 token 数（`num_computed_tokens`）；本步新调度的 token 数，$N_{\mathrm{new}}=N_{\mathrm{target}}-N_{\mathrm{computed}}$ | ch04、ch08 |
| $N_{\mathrm{placeholders}}$、$N_{\mathrm{in\ flight}}$、$N_{\mathrm{confirmed}}$、$N_{\mathrm{cacheable}}$ | async 调度里的占位 token、在途 token、已确认 token、可写入 prefix cache 的 token | ch04、ch08 |
| $N_{\mathrm{sampled}}$ | 一步里正常采样得到的 token 数（生成式模型为 1，pooling 与 diffusion 为 0；`num_sampled_tokens_per_step`，`vllm/v1/core/sched/scheduler.py:152`） | ch08 §8.3.2 |
| $N_{\mathrm{scheduled}}$、$N_{\mathrm{remaining}}$、$N_{\mathrm{chunk}}$ | 本步调度给请求的 token 数；还剩多少 token 没算；chunked prefill 本次切下的 token 数 | ch04 |
| $N_{\mathrm{budget}}$、$N_{\mathrm{budget,in}}$、$N_{\mathrm{max}}$ | 本步剩余的 token 预算；输入侧预算；`max_tokens` | ch04 |
| $N_{\mathrm{hit,local}}$、$N_{\mathrm{hit,ext}}$ | 本地、外部前缀命中的 token 数 | ch04 |
| $N_{\mathrm{ret}}$、$N_{\mathrm{look}}$、$N_{\mathrm{tail}}$ | 滑窗保留的 token 数；lookahead token 数；最后一个 block 的尾部空位 | ch05、ch06 |
| $N_{\mathrm{mm}}$ | 一个多模态 item 占的 encoder token 数 | ch04 §4.16 |
| $N_{q,r}$、$N_{\mathrm{ctx},r}$ | 请求 $r$ 本步的 query token 数；它已有的上下文 token 数，$N_{\mathrm{ctx},r}=S_r-N_{q,r}$ | ch06 |
| $N_{\mathrm{row},i}$、$N_{\mathrm{rt},i}$、$N_r$ | logits 行数；路由到 expert rank $i$ 的 (token, expert) 对数（每个 token 选 $\mathtt{top\_k}$ 个 expert，各 rank 的对数之和是 $N_{\mathrm{tok}}\times\mathtt{top\_k}$；这里的 `top_k` 是 MoE 路由参数，不是采样的 $\mathrm{top\_k}$）；PCP rank $r$ 的真实 token 数 | ch09 |
| $N_{\mathrm{in}}^{\Sigma}$、$N_{\mathrm{out}}^{\Sigma}$ | benchmark 里成功请求的输入 token 总数、输出 token 总数 | ch10 |
| $n_{\mathrm{null}}$、$n_{\mathrm{chunk}}$ | 被换成 null block 的 block 数；kernel 一次写入要切成的 chunk 数 | ch05、ch06 |
| $n_{\mathrm{step}}$、$n_{\mathrm{phase}}$、$n_{\mathrm{grp}}$ | step 计数（ch04 是 EngineCore 的步计数，ch08 教学例子里是连续跑的 step 数）；一次集合通信的阶段数；all-reduce 的分组数 | ch04、ch08、ch09 |
| $n_{\mathrm{wait},i}$、$n_{\mathrm{run},i}$、$n_{\mathrm{fly},i}$、$n_{\mathrm{cli}}$ | engine $i$ 的 waiting、running 请求数；client 观察到的 in-flight 数；client 数 | ch09 §9.7.3 |
| $n_{\mathrm{req}}$、$n_{\mathrm{SLO}}$、$n_{\mathrm{users}}$、$n_{\mathrm{GPU}}$ | benchmark 的成功请求数、达标请求数、并发用户数、GPU 数 | ch09、ch10 |
| $n_{\mathrm{sys}}$、$n_{\mathrm{queue}}$ | Little 定律里系统内、队列内的平均请求数 | ch10 |

位置与地址：

| 符号 | 含义 | 备注 |
|---|---|---|
| $\mathrm{pos}$ | token 在序列里的位置（从 0 起） | 旧 $p$ |
| $\mathrm{off}$ | token 在 block 内的偏移，$\mathrm{off}=\mathrm{pos}\bmod B_{\mathrm{blk}}$ | 旧 $o$ |
| $\mathrm{slot}$ | `slot_mapping` 里的槽位号，$\mathrm{slot}=b\,B_{\mathrm{blk}}+\mathrm{off}$ | 其中 $b$ 是物理 block ID，只是 ch06 §6.4 的局部字母（第 3 节） |
| $\mathrm{qsl}_r$ | `query_start_loc[r]`，请求 $r$ 的 query 在展平 batch 里的起点 | 旧 $q_r$ |

逻辑 block 序号 $i$、物理 block ID $b$ 都是 ch06 §6.4 的局部字母，不是全局符号（$b$ 另与 ch09 的偏置 $b_i$、ch10 的 burstiness $b$ 各自局部）。

下标里的 $r$ 在不同位置指不同的东西：$N_{q,r}$、$N_{\mathrm{ctx},r}$、$S_r$、$\mathrm{qsl}_r$ 的 $r$ 是请求序号；ch09 §9.9.3 的 $N_r$、$\mathrm{pad}_{r,t}$ 的 $r$ 是 PCP rank（§9.9.2 里 PCP rank 写 $i_{\mathrm{pcp}}$，就是为了不和请求序号 $r$ 混淆）。同一个公式里不会同时出现两种 $r$。

### 1.4 字节与显存：$b$ 与 $M$

| 符号 | 含义 | 单位 | 备注 |
|---|---|---|---|
| $b_{\mathrm{elem}}$ | 一个元素占的字节数（bf16 为 2） | byte/元素 | 旧 $B_{\mathrm{dtype}}$ |
| $M_{\mathrm{tok}}$ | 每 token 的 KV 字节数，$M_{\mathrm{tok}}=2LH_{\mathrm{kv}}d_hb_{\mathrm{elem}}$（单卡用 $H_{\mathrm{kv,local}}$） | byte/token | 旧 $B_{\mathrm{token}}$（ch01、ch05）；ch04 同名的 token 预算改为 $N_{\mathrm{budget}}$。ch01 L906 先给通式 $M_{\mathrm{tok}}=L(H_kd_k+H_vd_v)\,b_{\mathrm{elem}}$（K、V 的 head 数和宽度可以不同），本行是 $H_k=H_v=H_{\mathrm{kv}}$、$d_k=d_v=d_h$ 的特例；MLA 等压缩缓存不在这两个式子之内 |
| $M_{\mathrm{blk}}$、$M_{\mathrm{seq}}$ | 每 block、每序列的 KV 字节数：$M_{\mathrm{blk}}=B_{\mathrm{blk}}M_{\mathrm{tok}}$，$M_{\mathrm{seq}}=S\,M_{\mathrm{tok}}$ | byte | 旧 $B_{\mathrm{block}}$、$B_{\mathrm{sequence}}$ |
| $M_{\mathrm{total}}$、$M_{2048}$ | 全部请求的 KV 总字节数；2048 token 的 KV 字节数（ch01 算例） | byte | 旧 $B_{\mathrm{total}}$、$B_{2048}$ |
| $M_{\mathrm{KV}}$ | 可分给 KV Cache 的显存，$M_{\mathrm{KV}}=M_{\mathrm{req}}-M_{\mathrm{nonKV}}-M_{\mathrm{graph}}$ | byte | ch07 §7.9 与 ch08 §8.17.2 讲的是同一个量 |
| $M_{\mathrm{req}}$、$M_{\mathrm{nonKV}}$、$M_{\mathrm{graph}}$ | 请求的显存预算（`requested_memory`）；非 KV 峰值；从预算里扣除的 CUDA Graph 估算（`cudagraph_memory_estimate`） | byte | 旧 $M_{\mathrm{requested}}$、$M_{\mathrm{graph,applied}}$、$M_{\mathrm{graph\ reserve}}$；ch07 里的 `M_util`、`M_external` 是代码样式的中间量，保持不变 |
| $M_{\mathrm{payload}}$ | 一次通信的负载字节数 | byte | 旧 $B_{\mathrm{payload}}$ |

### 1.5 并行度与排名

| 符号 | 含义 | 备注 |
|---|---|---|
| $\mathrm{TP}$、$\mathrm{PP}$、$\mathrm{DP}$、$\mathrm{EP}$、$\mathrm{PCP}$、$\mathrm{DCP}$ | 各种并行的路数（正体，不用 `p`、`P`、`N`） | 旧 ch09 的 $p$（TP 路数）、$P$（PCP 大小）、$N$（DP group 大小）；ch03、ch08 的 `P`（PP stage 数）改 $\mathrm{PP}$ |
| $W_{\mathrm{engine}}$、$W_{\mathrm{all\_dp}}$ | world size：一个 DP engine 的 Worker 数 $W_{\mathrm{engine}}=\mathrm{TP}\times\mathrm{PP}\times\mathrm{PCP}$；所有 DP engine 合起来 $W_{\mathrm{all\_dp}}=W_{\mathrm{engine}}\times\mathrm{DP}$ | 单独的 $W$ 不再使用（旧 ch09 的 $W$、$W_i$ 表示 waiting 数，现在写 $n_{\mathrm{wait},i}$） |
| $g$、$\ell$、$i_{\mathrm{dp}}$ | 本机逻辑 GPU 编号（Worker 绑定 `cuda:g`）、local rank（Worker 在自己所属 DP engine 内的编号，$0\le\ell<W_{\mathrm{engine}}$）、DP engine 在本机上的序号，$g=\ell+i_{\mathrm{dp}}W_{\mathrm{engine}}$ | 旧 $d$；$g$ **不是**全局 rank；$\ell$ 只表示 local rank，ch06 在线 softmax 的分母改写 $Z$ |
| $i_{\mathrm{pp}}$、$i_{\mathrm{pcp}}$、$i_{\mathrm{tp}}$ | Worker 在 PP、PCP、TP 轴上的坐标（从 0 起，分别小于 $\mathrm{PP}$、$\mathrm{PCP}$、$\mathrm{TP}$），与 $i_{\mathrm{dp}}$ 同族 | ch09 §9.3.3 的全局 rank 公式；§9.9.2 起用 $i_{\mathrm{pcp}}$ 表示 PCP rank |
| $\mathrm{rank}$、$\mathrm{rank}_{\mathrm{out}}$ | 全局 rank（无 ExternalDP 时，由 $i_{\mathrm{dp}}$、$i_{\mathrm{pp}}$、$i_{\mathrm{pcp}}$、$i_{\mathrm{tp}}$ 算出）；一个 engine 内负责回复的 Worker 的 rank（`output_rank`） | ch09 §9.3.3、§9.6.6、§9.10.3；不是 DP 全局 rank |
| $r_{\mathrm{ep}}$ | 本卡在 EP 组里的编号（`ep_rank`），$0\le r_{\mathrm{ep}}<\mathrm{EP}$ | ch09 §9.8 |
| $I_{\mathrm{il}}$ | DCP 的 KV interleave 大小（`cp_kv_cache_interleave_size`） | 旧 $I$（ch09）；$I_{\mathrm{pf}}$ 是 ch04 的 `prefill_schedule_interval`（第 3 节局部字母），两者是不同的量 |

### 1.6 时间、吞吐与排队

| 符号 | 含义 | 单位 | 备注 |
|---|---|---|---|
| $\Delta t$ | 一段耗时的通式，下标说明是哪一段；ch10 §10.7.3 的请求间隔直接写 $\Delta t$ | s 或 ms | 旧 $T_{\mathrm{xxx}}$、`T`；下面几行是已登记的全部 $\Delta t_{\dots}$。单位随章：ch08 的串行教学模型用 ms，ch09 在算例里标明 s 或 ms，ch10 的指标公式用 s；每个公式的“其中”写明单位 |
| $\Delta t_{\mathrm{step}}$、$\Delta t_{\mathrm{sched}}$、$\Delta t_{\mathrm{prepare}}$、$\Delta t_{\mathrm{H2D}}$、$\Delta t_{\mathrm{forward}}$、$\Delta t_{\mathrm{sample}}$、$\Delta t_{\mathrm{D2H}}$、$\Delta t_{\mathrm{update}}$ | ch08 串行教学模型里一个 step 的总耗时与各段耗时 | ms | 旧 $T_{\dots}$ |
| $\Delta t_{\mathrm{layer}}$、$\Delta t_{\mathrm{compute}}$、$\Delta t_{\mathrm{collective}}$、$\Delta t_{\mathrm{imbalance}}$、$\Delta t_{\mathrm{comm}}$ | ch09 的 TP 单层耗时及其计算、集合通信、负载不均三项；一次通信的耗时 | s 或 ms | 旧 $T_{\dots}$；§9.12.1 比较串行与重叠时写 $\Delta t_{\mathrm{layer}}^{\mathrm{serial}}$、$\Delta t_{\mathrm{layer}}^{\mathrm{ov}}$ |
| $\Delta t_{\mathrm{EP}}$、$\Delta t_{\mathrm{dispatch}}$、$\Delta t_{\mathrm{expert}}$、$\Delta t_{\mathrm{combine}}$ | ch09 的 EP 一层耗时及 dispatch、expert 计算、combine 三段 | s 或 ms | 旧 $T_{\dots}$ |
| $\Delta t_{\mathrm{client\ queue}}$、$\Delta t_{\mathrm{queue}}$、$\Delta t_{\mathrm{prefill}}$、$\Delta t_{\mathrm{decode}}$、$\Delta t_{\mathrm{inference}}$、$\Delta t_{\mathrm{bench}}$ | ch10 一个请求的客户端排队、服务端排队、prefill、decode、inference 耗时；一次 benchmark 的时长 | s 或 ms | 旧 $T_{\dots}$；$\Delta t_{\mathrm{queue}}$ 只表示**单个请求**的排队等待（§10.11.1） |
| $\Delta t_{\mathrm{silent}}$、$\Delta t_{\mathrm{timeout}}$ | ch03 看门狗：距上次 `feed()` 已经过的时间；`WatchdogConfig.timeout` | s | |
| $\overline{\Delta t}_{\mathrm{sys}}$、$\overline{\Delta t}_{\mathrm{wait}}$ | Little 定律里请求在系统内、在队列里的**平均**停留时间 | s | 旧 $W$、$W_q$；带横线表示平均，与单个请求的 $\Delta t_{\mathrm{queue}}$ 是两个量 |
| $t_{\mathrm{arrival}}$、$t_{\mathrm{send}}$、$t_{\mathrm{first}}$、$t_{\mathrm{last}}$、$t_j$ | 客户端（load generator）的时间戳：请求到达（创建任务）、发送请求、收到第一个 token、收到最后一个 token、收到第 $j$ 个 token | s | 指标的定义靠时间戳相减：$\mathrm{TTFT}=t_{\mathrm{first}}-t_{\mathrm{send}}$ |
| $t_{\mathrm{queued,first}}$、$t_{\mathrm{scheduled,first}}$、$t_{\mathrm{token,first}}$、$t_{\mathrm{token,last}}$ | 服务端（EngineCore 单调时钟）记录的时间戳：第一次进入队列、第一次被调度、产生第一个 token、产生最后一个 token（对应 `queued_ts`、`scheduled_ts`、`first_token_ts`、`last_token_ts`） | s | ch10 §10.11 |
| $t_{\mathrm{now}}$、$t_{\mathrm{feed}}$、$t_{\mathrm{log}}$ | ch03 看门狗：检查时 `time.monotonic()` 的读数；最近一次 `feed()` 的时刻（`last_feed_time`）；上次转储的时刻（`last_timeout_log`，初值 0） | s | 时间戳，相减得 $\Delta t_{\mathrm{silent}}$ |
| $\mathrm{TTFT}$、$\mathrm{ITL}$、$\mathrm{TPOT}$、$\mathrm{E2EL}$ | 首 token 时延、token 间隔、每输出 token 耗时、端到端时延 | s（公式与算例）；vLLM 输出字段和命令行阈值用 ms，换算时除以 1000 | 正体；$\mathrm{SLO}_{\mathrm{ttft}}$ 等是对应的阈值，旧 $S_{\mathrm{ttft}}$；ch10 的指标公式按秒写，遇到带 `_ms` 的字段或毫秒阈值先换算 |
| $\lambda$、$\rho$ | 请求到达率（request/s）；利用率（0 到 1），也用于 KV 使用率 $\rho_{\mathrm{kv},i}$ | | |
| $\mathrm{thr}_{\mathrm{req}}$、$\mathrm{thr}_{\mathrm{tot}}$、$\mathrm{thr}_{\mathrm{out}}$、$\mathrm{thr}_{\mathrm{user}}$、$\mathrm{thr}_{\mathrm{GPU}}$、$\mathrm{thr}(n_{\mathrm{GPU}})$ | 吞吐：请求吞吐（request/s）、总 token 吞吐（输入加输出，token/s）、输出吞吐（token/s）、每用户（token/(s·user)）、每 GPU（token/(s·GPU)）；$\mathrm{thr}(n_{\mathrm{GPU}})$ 是 $n_{\mathrm{GPU}}$ 张卡时的吞吐 | | 旧 $Q_{\mathrm{out}}$、$Q_{\mathrm{user}}$、$Q_{\mathrm{GPU}}$、$Q_p$、$Q_1$；$Q$ 留给 Query 矩阵 |
| $G$ | goodput（request/s），$G=n_{\mathrm{SLO}}/\Delta t_{\mathrm{bench}}$ | | |
| $\mathrm{speedup}$、$\mathrm{efficiency}$ | 加速比 $\mathrm{thr}(n_{\mathrm{GPU}})/\mathrm{thr}(1)$；扩展效率 $\mathrm{speedup}/n_{\mathrm{GPU}}$ | | ch10 的 Amdahl 上限写 $\mathrm{speedup}_{\mathrm{total}}$，旧 $S_{\mathrm{total}}$ |

## 2 命名规则

1. **一字母一义。** 第 1 节出现过的符号（连同下标，如 $b_{\mathrm{elem}}$），在全书任何位置都只表示表里的那个量；同一个符号在第 1 节里只登记一次。
2. **描述性下标一律正体**：`B_{\mathrm{blk}}`、`N_{\mathrm{computed}}`、`H_{\mathrm{kv}}`；索引（$i$、$j$、$r$、$t$）和单个字母的助记下标
   （$d_h$、$H_q$、$N_q$、$W_Q$：head、query、Query 矩阵）用斜体。
3. **前缀表示量纲类别**：$N_{\dots}$ token 数、$n_{\dots}$ 其他计数、$M_{\dots}$ 字节、$b_{\dots}$ 每元素字节、
   $\Delta t_{\dots}$ 耗时、$t_{\dots}$ 时间戳、$B_{\dots}$ 只用于 block 大小。
4. **并行度和指标名用正体**（$\mathrm{TP}$、$\mathrm{TTFT}$），函数名用 `\operatorname`（$\operatorname{softmax}$、$\operatorname{LSE}$）。
5. **代码名与数学记号并列**：第一次出现时写 `num_computed_tokens`（$N_{\mathrm{computed}}$），之后在公式里用记号，
   在讲源码时用代码名。
6. **同一个量只有一种写法**：同一章里已经有的写法优先；跨章不一致的，以第 1 节为准。
7. **局部字母**只在所在小节有效，必须不在第 1 节（整个符号相同才算冲突，$b$ 与 $b_{\mathrm{elem}}$ 是两个符号），并在首次出现处用“其中”定义，在第 3 节登记；
   不同章的局部字母可以重名。第 1 节与第 3 节的符号互不相交由 `tests/test_notation.py` 检查。
8. **索引**（$i$、$j$、$k$、$r$、$t$）一律是局部字母，不登记在第 1 节；第 1 节里 $z_i$、$N_{q,r}$ 的 $i$、$r$ 只是下标。

## 3 局部字母表

下面这些符号在各自的小节里有局部含义，不会跨小节沿用；它们不在第 1 节，因此不构成冲突（规则 7，有测试检查）。
每个符号单独写成一个 `$…$`，同一格里用顿号分开；“其中”里的定义写在首次出现处。

| 章 | 局部符号 | 含义 |
|---|---|---|
| ch01 | $x_0,x_1$、$t$、$K_0,K_1,K_2$、$V_0,V_1,V_2$ | RoPE 配对的两个分量（旋转角用全局的 $\theta$）；token 序号；单头例子里 K、V 的行 |
| ch01 | $x$、$u$、$h$、$c$、$\omega$、$\varphi$、$R$、$\theta_1,\theta_2$ | 每个符号只在定义它的那一节有效：MLP 里一个 token 的列向量；SiLU 的标量输入；Query head 序号；softmax 数值稳定时减去的常数；RoPE 第 $j$ 对分量的“每个位置转多少”；点的原有角度；$2\times2$ 旋转矩阵 $R(\theta)$；证明相对位置性质时的任意两个角（RoPE 旋转角固定用全局的 $\theta$） |
| ch01 | $i_{\mathrm{next}}$、$\mathrm{kept}$、$\mathrm{do\_top\_p}$、$\mathrm{has\_topp\_only\_rows}$、$p_a$、$p_b$、$z_a$、$z_b$ | greedy 选出的 token ID；top-k/top-p 之后保留的候选集合；源码布尔变量 `do_top_p`、`has_topp_only_rows` 的公式写法；任意两个候选 $a$、$b$ 的概率与 logit（即全局的 $p_i$、$z_i$ 取 $i=a,b$） |
| ch01、ch02 | `n`（代码样式） | `SamplingParams.n`，要采样的条数，是 API 参数，不是公式量 |
| ch03 | $\mathrm{dump}$ | 看门狗这一轮检查要不要转储线程栈（布尔量）；`max_concurrent_batches` 是源码属性名，直接写成 $\mathtt{max\_concurrent\_batches}$，不另造符号 |
| ch04 | $\delta$、$k$、$j$、$r$、$a_{\mathrm{async}}$、$C_{\mathrm{step}}$、$C_{\mathrm{threshold}}$、$C_{\mathrm{other}}$、$I_{\mathrm{pf}}$、$\mathrm{pos}_{\mathrm{mm}}$、$\mathrm{cap\_bound}$ | §4.16.1 的 lookahead 偏移；§4.19.3 一次返回的 token 数与触发 stop 的序号；请求下标；async 在途采样步标志（0 或 1）；token 上限；`prefill_schedule_interval`；多模态 item 起点位置（`start_pos`）；`prefill_capacity_bound` 布尔量（旧 $B$）。带撇号的 $N_{\mathrm{new}}'$、$N_{\mathrm{computed}}'$、$N_{\mathrm{placeholders}}'$ 是对应量更新后的值，不单独登记 |
| ch05 | $E_i$、$i$、$j$ | 覆盖 block 与 lookahead 范围的 `extra_keys`；block 序号 |
| ch06 | $i$、$b$、$o$、$m$、$Z$、$m'$、$Z'$、$m_{\mathrm{tile}}$、$\mathrm{pos}_0$、$E_{\mathrm{max}}$、$B_{\mathrm{mgr}}$、$B_{\mathrm{ker}}$、$\kappa$、$i_{\mathrm{mgr}}$ | §6.4 地址公式里的逻辑 block 序号与物理 block ID；$o$ 是 attention 的输出向量；$m,Z$ 是在线 softmax 的当前最大值与分母（旧 $\ell$，让给 ch09 的 local rank），带撇号是合入新片段之后的值，$m_{\mathrm{tile}}$ 是新片段的最大分数；$\mathrm{pos}_0$ 是滑窗里首个 query 的位置；$E_{\mathrm{max}}$ 是一次 staged 写入的最长元素数；manager、kernel 的 block 大小、两者之比 $\kappa=B_{\mathrm{mgr}}/B_{\mathrm{ker}}$ 与 manager block 序号 |
| ch07 | $k$、$x$ | FLy 判定里熵的 top-$k$ 大小（不是采样的 top-k）；输入 |
| ch08 | $N_{\mathrm{graph}}$、$N_{\mathrm{pad}}$、$N_{\mathrm{uni}}$、$n_{\mathrm{ub}}$、$R$、$R_{\mathrm{graph}}$ | §8.14 全节（8.14.1–8.14.7）：descriptor 的 `num_tokens`（token，padding 之后）；padding 补的行数，$N_{\mathrm{pad}}=N_{\mathrm{graph}}-T$（token）；均匀 decode 批里每个请求的 query token 数（token/请求，批不是均匀 decode 时为 $\varnothing$）；批被切成的微批数（`num_ubatches`，个）；一批的请求数；descriptor 的 `num_reqs`（请求）。这里的 $R$ 与第 1 节的 $B$（教学形状里的批大小）、$n_{\mathrm{req}}$（benchmark 的成功请求数）不是同一个量 |
| ch08 | $\varnothing$、$\mathrm{compat}$ | §8.14.2：空值（源码的 `None`）；六项条件的合取（布尔量） |
| ch08 | $m$、$R_{\mathrm{key}}$、$\mathrm{tbl}$、$j$、$t$、$N_{\mathrm{pick}}$、$\mathcal S_{\mathrm{mode}}$、$u_k$、$k$ | §8.14.3：批里最长的 query token 数 $m=\max_rN_{q,r}$；decode 表的查表请求数（源码变量 `n`）；decode 表这一组的列表及其下标；token 表的键（token 数）；键 $t$ 对应的“刚好够大”的 graph token 数；同一模式、同一 LoRA 情形下 token 键 descriptor 的 token 数集合；这个集合里从小到大第 $k$ 个 token 数 |
| ch08 | $N_{\mathrm{buf}}$、$M_{\mathrm{buf}}$ | §8.14.5：持久输出 buffer 的行数（token）与字节数（byte） |
| ch08 | $N_{\mathrm{cap}}$、$N_{\mathrm{dec,cap}}$、$N_{\mathrm{dec,budget}}$、$N_{\mathrm{shared}}$ | §8.14.7：记录下来的最大 capture token 数（`max_cudagraph_capture_size`）；decode graph 的 token 上限；decode 的 token 预算（`max_decode_cudagraph_capture_tokens`）；拆分之前的共享上限（token） |
| ch08 | $R_{\mathrm{lim}}$、$R_{\mathrm{uni,max}}$、$R_{\max}$、$R_j$、$R_{j^\ast}$、$T_j$ | §8.14.7（请求或 token）：一个 query 长度下 decode graph 的请求数上限；这个 query 长度的批最多有的请求数；`max_num_seqs`；decode 请求数列表里第 $j$ 个数；第一个不小于 $R_{\mathrm{lim}}$ 的元素；共享 token 列表里第 $j$ 个数 |
| ch08 | $\mathcal L_{\mathrm{dec}}$、$\mathcal L_{\mathrm{pre}}$、$\mathcal L_{\mathrm{shared}}$、$\mathcal E$、$\mathcal Q$ | §8.14.7：decode 的请求数列表；prefill 的 token 数列表；共享的旧 token 列表；一个 query 长度要捕获的（请求数, query 长度）对的集合；target 的 decode query 长度集合 |
| ch08 | $x$、$x_{\mathrm{next}}$、$\mathrm{step}$、$\mathrm{stride}$、$n_{\mathrm{desc}}$、$n_{\mathrm{lora}}$ | §8.14.7：网格里当前的数及其下一个数；网格函数里“想要的步长”与取成 2 的幂后的步长；要捕获的 descriptor 总数；LoRA 情形数（`len(lora_capture_cases)`） |
| ch09 | $A,A_i,X_i,Y_i,b_i$、$Y^{\mathrm{part}}_i$、$u_r,f_r,\Sigma_u,\Sigma_f$、$G_2$、$\hat T$、$\mathrm{pad}_{r,t}$、$\alpha,\beta$、$N_{\mathrm{sch},i}$、$N_{\mathrm{drf},i}$、$N_{\mathrm{acc},i}$ | TP 线性层的权重与分片（$A$ 是权重，为避免与 world size $W$ 冲突）、偏置 $b_i$；行并行的 partial output；DP 同步的两个标志和它们的和；all-reduce 分组大小；PCP 填充后每 rank 的 token 数及其掩码；通信的固定延迟（s）与有效带宽（byte/s）（ch01 的 RoPE 角用 $\theta_1$、$\theta_2$）；「v0.33.0：batch-sharded sampling 可与自适应验证同开」一节里请求 $i$ 调度时预留的 logits 行数、带的 draft token 数、实际接纳的 draft 数（$N_{\mathrm{sch},i}$ 是按调度上限预留的 logits 行数，**不等于**第 1.3 节的 $N_{\mathrm{scheduled}}$，后者是 ch04 里请求本步被调度的 token 数） |
| ch09 | $\mathrm{score}_i$、$\mathrm{score}^{\mathrm{base}}_i$ | §9.7.3 engine $i$ 的负载分数及其基础分（无单位，分数低的 engine 优先接新请求） |
| ch09 | $n_{\mathrm{exp}}$、$n_{\mathrm{base}}$、$n_{\mathrm{rem}}$、$n_{\mathrm{exp},r}$、$\mathrm{start}_r$ | §9.8.2 expert 放置：expert 总数、每个 rank 至少拥有的个数、多出来的个数、rank $r_{\mathrm{ep}}$ 实际拥有的个数、`linear` 放置时它的起点 |
| ch09 | $t$、$k$、$x_t$、$y_t$、$e_{t,k}$、$w_{t,k}$、$E_e$、$N_{\mathrm{tok}}$ | §9.8.3 MoE 一层：token 序号、该 token 的第 $k$ 个选中 expert、输入与输出、选中的 expert 编号与路由权重、第 $e$ 个 expert 的 MLP、这一层要处理的 token 数 |
| ch09 | $c$、$\mathcal C_{i_{\mathrm{pcp}}}$、$\mathrm{work}$ | §9.9.2 PCP：每个 chunk 的 token 数、rank $i_{\mathrm{pcp}}$ 拿到的 chunk 编号集合、causal 工作量（(query, key) 对数） |
| ch09 | $\mathrm{local}$、$r_{\mathrm{dcp}}$、$S_{\mathrm{loc},r_{\mathrm{dcp}}}$、$n_{\mathrm{rnd}}$、$n_{\mathrm{rest}}$、$S_{\max}$、$n_{\mathrm{blk}}^{\max}$ | §9.9.4、§9.9.7 DCP：位置 $\mathrm{pos}$ 的 token 在它的 owner rank 上的本地位置 $\mathrm{local}(\mathrm{pos})$；DCP rank 序号；长度为 $S$ 的序列（$S$ 用全局符号）里该 rank 本地存的 token 数；把 $S$ 按 $I_{\mathrm{il}}\mathrm{DCP}$ 拆成的整轮数与余数（§9.9.7 推上界用）；`max_model_len`（token，序列长度的上限）；一个请求在每个 rank 上最多占的 block 数 |
| ch09 | $\mathcal K_i$、$Z_i$、$v_j$、$w_i$ | §9.9.5 partial attention：KV 分片 $i$ 持有的 key 集合、它的 softmax 分母 $Z_i=\sum_{j\in\mathcal K_i}e^{s_j}$、value 行、合并权重 $w_i=e^{\operatorname{LSE}_i-\operatorname{LSE}}$；$s_j$ 用全局的（固定一个 query 对 key $j$ 的打分） |
| ch09 | $n_{\mathrm{pack}}$ | §9.9.6 all-to-all 缓冲区里每个 head 为打包 LSE 额外占用的元素数（bf16 时为 2，fp32 输出时为 1） |
| ch09 | $M_{\mathrm{avail},g}$、$M_{\mathrm{avail}}$ | §9.10.5 rank $g$ 本地测得的可用 KV 显存与对齐后的最小值（byte） |
| ch09 | $n_{\mathrm{node}}$、$i_{\mathrm{node}}$、$W_{\mathrm{local}}$、$r_{\mathrm{eng}}$ | §9.11.2 多节点：一个 DP engine 跨的节点数、本节点的序号、每个节点上的 Worker 数（`local_world_size`）、Worker 在 engine 内的全局 rank（不是 §9.2.1 的 $g$） |
| ch10 | $b$、$f$、$s$、$x_{\mathrm{treatment}}$、$x_{\mathrm{baseline}}$、$\Delta_{\%}$ | 到达间隔 Gamma 分布的 burstiness；Amdahl 公式里被加速部分的占比与加速倍数；对比实验的指标值与相对变化 |
| ch10 | $\mu$、$P_k$、$\sigma$、$\Phi$ | 本章局部，只在各自的小节里用：M/M/1 的服务率（request/s）与稳态时系统里恰有 $k$ 个请求的概率（§10.13.2）；单次测量相对均值的标准差与标准正态累积分布函数（§10.21.4） |

## 4 改名对照表

按章列出，每行是一类写法：旧写法、改成什么、为什么。各章正文已经按此表改写；改完的行不会从表里删除，
它的用处是认出旧资料和旧版本里的老写法，查找请直接用符号。

覆盖范围是 `$…$`、`$$…$$`、行内代码和 mermaid 标签；夹在中文句子里、没有写成公式或代码的字母（例如“块大小 B”）不在表内。

### ch01 Transformer 推理

| 旧写法 | 新写法 | 含义 |
|---|---|---|
| $T$（温度） | $\tau$ | softmax 温度 |
| $z_i/T$ | $z_i/\tau$ | softmax 温度 |
| $K^{\mathsf T}$ | $K^{\top}$ | 转置（同章后面另写作 `K^T`） |
| `K^T` | $K^{\top}$ | 转置 |
| $D_h$ / `D_h` / `Dh` | $d_h$（代码样式 `d_h`） | 单头宽度（Q/K） |
| $D_k$、$D_v$ / `Dv` | $d_k$、$d_v$ | K / V 的单头宽度（仅在两者可能不同的通式里出现） |
| `Hq`、`Hkv` | $H_q$、$H_{\mathrm{kv}}$（形状里 `H_q`、`H_kv`） | Q / KV head 数 |
| `Nq`、`Nk` | $N_q$、$N_k$ | 单头例子里 query / key 行数 |
| $M$（因果 mask） | $\mathrm{Mask}$ | 因果 mask（把 $M$ 留给显存/字节数） |
| 形状里的 `V`（词表） | `vocab_size` | 形状里的词表维（如 `[T, V]`、`[V]`、`[B,S,V]`） |
| $V$（词表） | $\lvert\mathcal V\rvert$ | 词表大小（同章后面 $V$ 是 Value 矩阵） |
| $k_i$、$k$（每行 top-k） | $\mathrm{top\_k}_i$、$\mathrm{top\_k}$ | 第 $i$ 行的 top-k（避免与 key $k$ 混淆） |
| $p_i$、$p$（每行 top-p） | $\mathrm{top\_p}_i$、$\mathrm{top\_p}$ | 第 $i$ 行的 top-p（$p_i$ 只留给概率；同章另一处的 $p_i$ 是概率） |
| `I`（MLP 中间宽度） | $D_{\mathrm{ff}}$ | MLP 中间宽度 |
| $B_{\mathrm{token}}$ | $M_{\mathrm{tok}}$ | 每 token 的 KV 字节数 |
| $B_{\mathrm{dtype}}$ | $b_{\mathrm{elem}}$ | 每个元素的字节数 |
| $B_{2048}$、$B_{\mathrm{total}}$ | $M_{2048}$、$M_{\mathrm{total}}$ | 2048 token / 全部请求的 KV 总字节数 |
| `T`（温度）、`T>0` 等 | $\tau$（`τ>0` 等） | softmax 温度（同章 `[T,D]` 里的 `T` 是 token 数） |
| `k`、`p`（top-k、top-p 参数） | $\mathrm{top\_k}$、$\mathrm{top\_p}$ | top-k / top-p 参数 |
| `P=128`、`G=32` | $N_{\mathrm{prompt}}=128$、$N_{\mathrm{out}}=32$ | 最小实验的 prompt / 输出 token 数（`G` 留给 ch10 的 goodput） |
| 上标 `T`（转置） | 上标 $\top$ | 转置（`T` 留给 token 数） |

### ch03 EngineCore

| 旧写法 | 新写法 | 含义 |
|---|---|---|
| `P`（PP stage 数） | $\mathrm{PP}$ | pipeline stage 数 |

### ch04 Scheduler

| 旧写法 | 新写法 | 含义 |
|---|---|---|
| $B_{\mathrm{token}}$ | $N_{\mathrm{budget}}$ | 剩余 token 预算（原文自注「不是字节数」） |
| $B_{\mathrm{input}}$ | $N_{\mathrm{budget,in}}$ | 输入侧 token 预算 |
| $H_{\mathrm{local}}$、$H_{\mathrm{external}}$（及代码样式 `H_local`、`H_external`） | $N_{\mathrm{hit,local}}$、$N_{\mathrm{hit,ext}}$ | 本地 / 外部前缀命中 token 数 |
| $s$ | $\mathrm{pos}_{\mathrm{mm}}$ | 多模态 item 起点位置（`start_pos`） |
| $L$ | $N_{\mathrm{mm}}$ | item 占的 encoder token 数（避免与层数 $L$ 冲突） |
| $c$ | $N_{\mathrm{computed}}$ | 已计算 token 数（同章后面已是 $N_{\mathrm{computed}}$） |
| $n$、$n'$ | $N_{\mathrm{new}}$、$N_{\mathrm{new}}'$ | 本轮新调度 token 数 |
| $n$ | $N_{\mathrm{new}}$ | 本轮调度 token 数 |
| $T$ | $N_{\mathrm{tokens}}$ | 请求当前已知 token 总数（`num_tokens`；同章后面已是 $N_{\mathrm{tokens}}$） |
| $O$ | $N_{\mathrm{out}}$ | 已确认输出 token 数 |
| $a$ | $a_{\mathrm{async}}$ | async 在途采样步标志（0/1） |
| $M$ | $N_{\mathrm{max}}$ | `max_tokens` |
| $c$、$c'$ | $N_{\mathrm{computed}}$、$N_{\mathrm{computed}}'$ | 已计算 token 数 |
| $P$、$P'$ | $N_{\mathrm{placeholders}}$、$N_{\mathrm{placeholders}}'$ | `num_output_placeholders`（同章后面已是 $N_{\mathrm{placeholders}}$） |
| $I$ | $I_{\mathrm{pf}}$ | `prefill_schedule_interval` |
| $N$ | $n_{\mathrm{step}}$ | 步计数（无单位整数） |
| $B$ | $\mathrm{cap\_bound}$ | `prefill_capacity_bound` 布尔量（$B$ 只留给批大小） |
| `H_local`、`H_external` | $N_{\mathrm{hit,local}}$、$N_{\mathrm{hit,ext}}$ | 前缀命中 token 数 |
| `% I` | `% I_pf` | `prefill_schedule_interval` |
| 图中 `c`、`n`、`s`、`T`、`O`、`a` | `N_computed`、`N_new`、`pos_mm`、`N_tokens`、`N_out`、`a_async` | 判定节点里的符号（纯文本） |
| 图中 `N`、`I` | `n_step`、`I_pf` | prefill 节拍图里的符号（纯文本） |

### ch05 KV Cache

| 旧写法 | 新写法 | 含义 |
|---|---|---|
| $B_{\mathrm{token}}$ | $M_{\mathrm{tok}}$ | 每 token 的 KV 字节数 |
| $B_{\mathrm{sequence}}$ | $M_{\mathrm{seq}}$ | 每序列的 KV 字节数 |
| $B_{\mathrm{block}}$ | $M_{\mathrm{blk}}$ | 每 block 的 KV 字节数 |
| $B_{\mathrm{dtype}}$ | $b_{\mathrm{elem}}$ | 每个元素的字节数 |
| $D_{\mathrm{head}}$ | $d_h$ | 单头宽度 |
| `T`（block 大小） | $B_{\mathrm{blk}}$ | block 大小（token 数） |
| $T$、$W_{\mathrm{tail}}$ | $B_{\mathrm{blk}}$、$N_{\mathrm{tail}}$ | block 大小；最后一个 block 的尾部空位 token 数 |
| $T$ | $B_{\mathrm{blk}}$ | block 大小 |
| $R$ | $N_{\mathrm{ret}}$ | 滑窗保留 token 数（`num_retained_tokens`） |
| $p$ | $\mathrm{pos}$ | token 位置 |
| $H_i$ | $\eta_i$ | 第 $i$ 个 block 的链式 hash（图里写作 `hash_i`） |
| $\mathrm{tokens}_i^{+L}$、`L` | $\mathrm{tokens}_i^{+N_{\mathrm{look}}}$ | lookahead token 数（避免与层数 $L$ 冲突） |
| 图中 `tokens 0 到 B-1` 等 | `tokens 0 到 B_blk-1` 等 | mermaid 标签用纯文本 |
| `L`（lookahead） | $N_{\mathrm{look}}$ | lookahead token 数 |
| `R`、`T`、`p` | $N_{\mathrm{ret}}$、$B_{\mathrm{blk}}$、$\mathrm{pos}$ | 滑窗 null block 的算例（`n_null` 本身不改名） |
| 图中 `L = 0`、`L = num_prefill_lookahead_tokens` | `N_look = 0`、`N_look = num_prefill_lookahead_tokens` | lookahead（纯文本） |
| 图中 `p = …`、`R = …` | `pos = …`、`N_ret = …` | 滑窗算例图（纯文本） |

### ch06 PagedAttention

| 旧写法 | 新写法 | 含义 |
|---|---|---|
| $B$ | $B_{\mathrm{blk}}$ | block 大小 |
| $p$ | $\mathrm{pos}$ | token 位置 |
| $o$（block 内偏移） | $\mathrm{off}$ | block 内偏移（$o$ 留给 attention 输出向量） |
| $p_0$ | $\mathrm{pos}_0$ | 窗口里首个 query 的位置 |
| $R$ | $N_{\mathrm{ret}}$ | 滑窗保留 token 数 |
| $L_{\max}$ | $E_{\mathrm{max}}$ | 一次 staged 写入的最长元素数（避免与层数 $L$ 冲突） |
| $q_r$ | $\mathrm{qsl}_r$ | `query_start_loc[r]`（$q$ 留给 query 向量） |
| $L^Q_r$ | $N_{q,r}$ | 请求 $r$ 本步的 query token 数 |
| $L^{\mathrm{seq}}_r$、$L^{\mathrm{context}}_r$ | $S_r$、$N_{\mathrm{ctx},r}$ | 请求 $r$ 的可见序列长度；已有上下文 token 数 |
| $\sqrt d$ | $\sqrt{d_h}$ | 单头宽度 |
| $p_j$（attention 权重） | $a_j$ | attention 权重（$p_i$ 只留给概率） |
| `B=4` | $B_{\mathrm{blk}}=4$ | block 大小 |
| $\ell$、$\ell'$ | $Z$、$Z'$ | 在线 softmax 的归一化分母（$\ell$ 留给 ch09 的本机 rank） |
| `A`、`K`、`r=A/K`、`m` | $B_{\mathrm{mgr}}$、$B_{\mathrm{ker}}$、$\kappa=B_{\mathrm{mgr}}/B_{\mathrm{ker}}$、$i_{\mathrm{mgr}}$ | manager / kernel block 大小与换算（`A/K/r/m` 与矩阵 $A,K$、请求下标 $r$、在线 softmax 的 $m$ 冲突） |
| 图中 `b times B` | `b times B_blk` | 地址公式图（纯文本） |
| 图中 `n_null = floor(max(p0-R, 0) / B)` | `n_null = floor(max(pos_0-N_ret, 0) / B_blk)` | null block 图（纯文本） |

### ch07 模型执行

| 旧写法 | 新写法 | 含义 |
|---|---|---|
| $M_{\mathrm{requested}}$ | $M_{\mathrm{req}}$ | `requested_memory` |
| $M_{\mathrm{graph},\mathrm{applied}}$ | $M_{\mathrm{graph}}$ | `cudagraph_memory_estimate`（“applied”只是旧写法里的描述性下标） |
| $T$（温度） | $\tau$ | softmax 温度 |
| $H(x)$、$H$ | $\mathcal H(x)$、$\mathcal H$ | top-k 熵（$H$ 留给 head 数） |
| `T>0`、`T=0`、`T=1` | `τ>0`、`τ=0`、`τ=1` | softmax 温度 |

### ch08 compile / CUDA Graph

| 旧写法 | 新写法 | 含义 |
|---|---|---|
| $T_{\mathrm{step}}$、$T_{\mathrm{sched}}$ 等 | $\Delta t_{\mathrm{step}}$、$\Delta t_{\mathrm{sched}}$ 等 | 耗时 |
| $M_{\mathrm{available\ KV}}$、$M_{\mathrm{graph\ reserve}}$ | $M_{\mathrm{KV}}$、$M_{\mathrm{graph}}$ | 可给 KV 的显存、graph 预留 |
| $M_{\mathrm{requested}}$ | $M_{\mathrm{req}}$ | `requested_memory` |
| `P`（PP stage 数） | $\mathrm{PP}$ | pipeline stage 数 |
| 源码变量 `n`（`n = cdiv(num_tokens, uniform_token_count)`） | $R_{\mathrm{key}}$ | §8.14.3 decode 表的查表请求数 |

### ch09 分布式

| 旧写法 | 新写法 | 含义 |
|---|---|---|
| $d$（DP 序号） | $i_{\mathrm{dp}}$ | DP engine 序号 |
| $A_{p-1}$、$X_{p-1}$、`p` | $A_{\mathrm{TP}-1}$、$X_{\mathrm{TP}-1}$ | TP 路数（$p$ 只留给概率/位置的旧用法） |
| $P_i$、`P_i` | $Y^{\mathrm{part}}_i$ | 行并行下第 $i$ 个分片的 partial output |
| $T_{\mathrm{layer}}(p)$ 等 | $\Delta t_{\mathrm{layer}}(\mathrm{TP})$ 等 | TP 路数与耗时 |
| $n_i$ | $N_{\mathrm{row},i}$ | logits 行数 |
| $s_i$ | $N_{\mathrm{sch},i}$ | 本步调度的 token 数 |
| $d_i$ | $N_{\mathrm{drf},i}$ | draft token 数（避免与 $d$ 冲突） |
| $a_i$ | $N_{\mathrm{acc},i}$ | 被接受的 draft token 数 |
| $W_i$（waiting） | $n_{\mathrm{wait},i}$ | engine $i$ 的 waiting 请求数（$W$ 只指 world size） |
| $R_i$ | $n_{\mathrm{run},i}$ | engine $i$ 的 running 请求数 |
| $U_i$ | $\rho_{\mathrm{kv},i}$ | KV 使用率（0–1） |
| $C$ | $n_{\mathrm{cli}}$ | client 数 |
| $I_i$ | $n_{\mathrm{fly},i}$ | client 观察到的 engine $i$ in-flight 数 |
| $N$（DP group 大小） | $\mathrm{DP}$ | DP group 大小（`dp_group.size()`） |
| $S_0$ | $\Sigma_u$ | `has_unfinished` 之和 |
| $S_1$ | $\Sigma_f$ | 暂停标志之和 |
| $p_r$ | $f_r$ | rank $r$ 是否有待处理的暂停请求（0/1） |
| $B$（块大小） | $G_2$ | all-reduce 的 2 的幂分组大小 |
| $M=N/B$ | $n_{\mathrm{grp}}=\mathrm{DP}/G_2$ | 分组数 |
| $P$ | $\mathrm{PCP}$ | PCP 大小 |
| $m_{r,t}$ | $\mathrm{pad}_{r,t}$ | padding 掩码（$m$ 留给在线 softmax 的最大值） |
| $n_r$ | $N_r$ | rank $r$ 的真实 token 数 |
| $I$ | $I_{\mathrm{il}}$ | `cp_kv_cache_interleave_size` |
| $p$ | $\mathrm{pos}$ | token 位置 |
| $L$、$L_i$、`L_i` | $\mathrm{LSE}$、$\mathrm{LSE}_i$ | Log-Sum-Exp（$L$ 只指层数） |
| $n_i$ | $N_{\mathrm{rt},i}$ | 路由到 expert rank $i$ 的 token 数 |
| $T_{\mathrm{EP}}$、$T_{\mathrm{dispatch}}$ 等 | $\Delta t_{\mathrm{EP}}$、$\Delta t_{\mathrm{dispatch}}$ 等 | 耗时 |
| $T_{\mathrm{comm}}$ | $\Delta t_{\mathrm{comm}}$ | 耗时 |
| $B_{\mathrm{payload}}$ | $M_{\mathrm{payload}}$ | 通信负载字节数 |
| `Q_1`、$Q_p$、`Q_p`、$p$（卡数） | $\mathrm{thr}(1)$、$\mathrm{thr}(n_{\mathrm{GPU}})$、$n_{\mathrm{GPU}}$ | $n_{\mathrm{GPU}}$ 卡时的吞吐与卡数（$Q$ 留给 Query 矩阵） |
| `Dout/p`、`Din/p` | `D_out/TP`、`D_in/TP` | TP 路数 |
| 图中 `u, p`、`B = N`、`M` | `u, f`、`G_2`（`DP` 的最大 2 的幂因子）、`n_grp` | DP all-reduce 图（纯文本） |

### ch10 实验方法

| 旧写法 | 新写法 | 含义 |
|---|---|---|
| `T`、$T$（benchmark 时长） | $\Delta t_{\mathrm{bench}}$ | benchmark 时长 |
| `R` | $n_{\mathrm{req}}$ | 成功请求数（请求数是计数，用 $n_{\dots}$） |
| `I` | $N_{\mathrm{in}}^{\Sigma}$ | 成功请求的输入 token 总数 |
| `O` | $N_{\mathrm{out}}^{\Sigma}$ | 成功请求的输出 token 总数 |
| $S_{\mathrm{ttft}}$、$S_{\mathrm{tpot}}$、$S_{\mathrm{e2el}}$ | $\mathrm{SLO}_{\mathrm{ttft}}$、$\mathrm{SLO}_{\mathrm{tpot}}$、$\mathrm{SLO}_{\mathrm{e2el}}$ | SLO 阈值 |
| $R_{\mathrm{SLO}}$ | $n_{\mathrm{SLO}}$ | 达标请求数 |
| $N$ | $N_{\mathrm{out}}$ | 单个请求的输出 token 数 |
| $T_{\mathrm{client\ queue}}$ | $\Delta t_{\mathrm{client\ queue}}$ | 客户端排队耗时 |
| $T_{\mathrm{queue}}$、$T_{\mathrm{prefill}}$、$T_{\mathrm{decode}}$、$T_{\mathrm{inference}}$ | $\Delta t_{\mathrm{queue}}$、$\Delta t_{\mathrm{prefill}}$、$\Delta t_{\mathrm{decode}}$、$\Delta t_{\mathrm{inference}}$ | 单个请求的四段耗时（$\Delta t_{\mathrm{queue}}$ 只指单个请求的排队等待） |
| $L$、$L_q$ | $n_{\mathrm{sys}}$、$n_{\mathrm{queue}}$ | 系统内 / 队列内平均请求数（$L$ 只指层数） |
| $W$、$W_q$ | $\overline{\Delta t}_{\mathrm{sys}}$、$\overline{\Delta t}_{\mathrm{wait}}$ | 系统内 / 队列内的**平均**停留时间（$W$ 只指 world size；带横线，与单个请求的 $\Delta t_{\mathrm{queue}}$ 区分） |
| $S_{\mathrm{total}}$ | $\mathrm{speedup}_{\mathrm{total}}$ | 端到端加速比（与 ch09 的 $\mathrm{speedup}$ 同族） |
| `N>1` | $N_{\mathrm{out}}>1$ | 输出 token 数 |
| $Q_{\mathrm{out}}$、$Q_{\mathrm{user}}$、$Q_{\mathrm{GPU}}$ | $\mathrm{thr}_{\mathrm{out}}$、$\mathrm{thr}_{\mathrm{user}}$、$\mathrm{thr}_{\mathrm{GPU}}$ | 吞吐（$Q$ 留给 Query 矩阵） |
| $N_{\mathrm{users}}$、$N_{\mathrm{GPU}}$ | $n_{\mathrm{users}}$、$n_{\mathrm{GPU}}$ | 用户数、GPU 数（计数，用 $n_{\dots}$；$N_{\dots}$ 的单位是 token） |

## 5 写法规则

1. 数学量写成 `$…$`；代码标识符、路径、参数名和程序输出写成代码样式，不把它们冒充数学。
2. 张量形状写成代码样式，维度名取自第 1 节：`[B,S,H_q,d_h]`、`[T,H_kv,d_h]`、`[T,D_in]`；一个形状里每个维度名只表示一个量。
3. 有量纲的量写单位并正体：`128\,\mathrm{KiB/token}`；乘法链里把单位一起带上做单位检查。
4. 描述性下标、并行度、指标名用 `\mathrm{…}`，函数名用 `\operatorname{…}`；舍入写 `\approx`；多步推导用 `aligned`。
5. mermaid 的标签不能用 LaTeX 或 HTML（`epub/mermaid-config.json` 里 `htmlLabels` 为 `false`），符号写成纯文本：
   `d_h`、`B_blk`、`M_tok`、`hash_i`、`tau`。
6. 公式前后的文字：首次出现时必须说明每个符号、shape、单位和这个公式回答的问题（`AGENTS.md` §8）；用第 6 节的模板写。
7. 第 1 节登记带固定含义的命名符号（可以只在一章里出现）；第 3 节登记只在一节里用的裸字母，并在正文里用“其中”定义。新增前者改第 1 节，新增后者改第 3 节。

## 6 公式讲解模板

每个公式（或紧密相关的一组公式）后面按下面四段写。**第三段“怎么算出来”是必写项**：必须给出推导
（从定义或源码规则一步步到公式），或者一个带数字的最小例子，从输入算到结果并做单位检查；两者至少要有一个。
只给公式和符号说明、不说明怎么算出来的，不算完成。

````markdown
**白话：**1–3 句，说这个公式在算什么、为什么这样算；不重复符号。

**其中：**每个符号的含义、shape、单位；第一次出现的局部字母在这里定义。

**怎么算出来：**
（推导）把公式从定义拆成因子，一步一个因子，说明每一步为什么乘/加这一项；
或（算例）取最小的具体数字，逐步代入，写出每一步的中间结果和最终单位。
定义式、源码规则则写出它对应的代码位置（`文件:行号` @ 固定 revision），再用一个算例走一遍。

**核对：**与源码、`examples/` 或测试的哪一处一致；能用脚本算的，写出命令和预期输出。
````

下面用 ch01 的 KV 字节数公式演示这个模板（数字与 ch01、ch05 的算例一致）：

$$
M_{\mathrm{tok}}=2\,L\,H_{\mathrm{kv}}\,d_h\,b_{\mathrm{elem}}
$$

**白话：**生成一个 token 要在每一层记下它的 K 和 V，这个公式数的是“一个 token 的 KV 要占多少字节”。

**其中：**$L$ 是层数；$H_{\mathrm{kv}}$ 是 KV head 数；$d_h$ 是单个 head 的宽度（元素）；$b_{\mathrm{elem}}$ 是每个元素的字节数；
前面的 $2$ 表示 K 和 V 两份。

**怎么算出来：**

1. 一层里，K 有 $H_{\mathrm{kv}}$ 个 head、每个 head $d_h$ 个元素，共 $H_{\mathrm{kv}}d_h$ 个元素；V 同样，所以一层一个 token 是 $2H_{\mathrm{kv}}d_h$ 个元素；
2. 共 $L$ 层，一个 token 是 $2LH_{\mathrm{kv}}d_h$ 个元素；
3. 每个元素 $b_{\mathrm{elem}}$ 字节，乘上去得到字节数。

代入 $L=32$、$H_{\mathrm{kv}}=8$、$d_h=128$、$b_{\mathrm{elem}}=2$（bf16）：

$$
2\times32\times8\times128\times2=131072\ \mathrm{byte/token}=128\ \mathrm{KiB/token}
$$

单位检查：元素数 $\times$ byte/元素 = byte，因为数的是“每个 token”，所以是 byte/token。

**核对：**与 ch01、ch05 里的 KV 字节数算例一致。

## 7 配图规则

目标是复杂的地方一定有图，但不多。

- **必配图的 10 处**（复杂到没有图就很难讲清）：
  ch01 单头 attention 数据流（§「Q、K、V 是什么」）、ch01 GQA 的 Q head 与 KV head 对应（§「Multi-Head 与 GQA」）；
  ch02 §2.15 五种队列的拓扑；ch03 §「v0.33.0：IPC 与共享内存不是同一个开关」；
  ch04 §4.19.3 speculative rejection 回滚的时间线；
  ch05 §5.5.3 多个 cache group 共享 block ID pool、§5.15.5 checkpoint 位置的三个分支；
  ch09 §9.2.1 三种 rank 与 GPU 编号、§9.9.5 partial attention 与 LSE 合并；
  ch10 §10.11.1 请求内部时间拆分。
- **其他位置**只在**确实复杂**时才配图：同时满足“涉及三个以上互相依赖的对象或状态，或一段文字在讲分支、顺序、坐标映射”
  与“读者没有图就很难复述主路径”。
- **每张图**（`AGENTS.md` §9）：只表达一个主要问题；节点能对应真实的模块、类或对象；关键边要能对到源码；
  标签用纯文本（第 5 节）；图后紧跟手写的读图方法引用块（格式同各章已有的图），不用通用文案；在本地用 mermaid-cli 渲染并看过 PNG（仓库不提交渲染产物）。

## 8 共享文件与维护约定

- 本文件（`glossary/notation.md`）是全书符号的唯一登记处：第 1 节是带固定含义的命名符号（可以只在一章里出现），
  第 3 节是只在一节里用的裸字母；改名后在第 4 节追加一行。第 1 节与第 3 节的符号互不相交、第 1 节的符号只登记一次，由 `tests/test_notation.py` 检查。
- `glossary/terms.md`、`book/README.md`、`book/reading-guide.md`：术语表、总览和阅读指南，改符号约定时同步检查。
- 章节保持 `draft`、`runtime_verified: false`；源码引用保持 pin `05caae03c20f69436ff7ec23ded339ea104f1a01`。
