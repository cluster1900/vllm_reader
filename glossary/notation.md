# 符号与记法约定

本文件回答一个问题：书里的每个字母到底指什么，换一章会不会变。全书约 400 个公式（行间与行内）原先
各自在附近定义符号，同一个字母在不同章里常常是不同的量（例如 `T` 先后表示 token 数、温度、block
大小和耗时），这里统一成一张表，并给出旧写法到新写法的对照。

使用方法：

- 读正文时遇到不确定的字母，先查第 1 节的全局符号表；表里没有的，是该小节自己定义的局部字母（第 3 节）。
- 写或改公式时，按第 2、5 节的规则选符号，按第 6 节的模板写讲解，按第 7 节决定要不要配图。
- 第 4 节是改名对照表：各章正文会按批次逐章改成新写法，改完之前，正文里看到的旧写法以这张表为准。
  表里的行号对应 vllm_reader main `14708f03`，批次开工时要重新核对。

这份约定是在一次全书普查之后定的，普查方法、数量和结果见
[符号普查记录](../meta/notation-census-2026-11-10.md)；重新普查用 `python3 scripts/notation_census.py`。
源码里的名字（如 `num_computed_tokens`）不受本约定约束，始终写成代码样式；公式里用的是它们的数学记号，
两者的对应关系在表里列出。

## 1 全局符号表

“全局”的意思是：在全书任何位置，这个字母（带这个下标）只表示表里的那一个量。
没有写单位的量是无单位的比例、下标或标志位。

### 1.1 形状与规模

| 符号 | 含义 | 单位 / shape | 备注 |
|---|---|---|---|
| $B$ | 批大小：一批里的请求（序列）数，只出现在教学形状 `[B,S,…]` 里 | 个 | block 大小另写 $B_{\mathrm{blk}}$，不再用 $B$ |
| $S$ | 单个请求的序列长度：该请求已有、需要存 KV 的 token 数；$S_r$ 是请求 $r$ 的 | token | ch01、ch05、ch06；与 ch04 的 $N_{\mathrm{tokens}}$ 是同一类量的两个视角（KV Cache 一侧、scheduler 一侧），各章保留原名 |
| $T$ | 一个 step 展平后的 token 总数，即 vLLM 形状 `[T,…]` 的第 0 维，$T=\sum_r N_{q,r}$ | token | 不再表示温度、block 大小或耗时 |
| $D$ | hidden size；$D_{\mathrm{in}}$、$D_{\mathrm{out}}$ 是某个线性层的输入、输出宽度；$D_{\mathrm{ff}}$ 是 MLP 中间宽度 | 元素 | 旧 `I`（MLP 宽度）见第 4 节 |
| $H_q$、$H_{\mathrm{kv}}$ | Query head 数、KV head 数；$H_{\mathrm{kv,local}}=H_{\mathrm{kv}}/\mathrm{TP}$ 是一张卡上的 KV head 数 | 个 | $H_k$、$H_v$ 只在 K、V head 数可能不同的通式里出现 |
| $d_h$ | 单个 head 的宽度（Query/Key，通常 Value 相同）；缩放因子写 $\sqrt{d_h}$ | 元素 | $D=H_q d_h$ 只在各 head 等宽时成立；$d_k$、$d_v$ 只在 K、V 宽度可能不同的通式里出现 |
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
| $O$ | attention 输出；$O_i$ 是分片 $i$ 的部分输出 | `[T,H_q·d_v]` | 不再表示输出 token 数 |
| $\mathrm{Mask}$ | causal mask（可见位置 0，不可见位置 $-\infty$） | `[N_q,N_k]` | 旧 $M$，把 $M$ 留给显存 |
| $s_j$、$a_j$ | 第 $j$ 个 key 的打分 $q\cdot k_j/\sqrt{d_h}$；softmax 之后的 attention 权重 $a_j$ | 无单位 | 旧 ch06 的 $p_j$；向量形式 $a$ 见 ch01 |
| $z_i$、$p_i$ | 候选 token $i$ 的 logit 与概率（softmax 之后，总和为 1） | 无单位 | $p_i$ 只表示概率 |
| $\tau$ | softmax 温度，$\tau>0$ | 无单位 | 旧 $T$（ch01、ch07） |
| $\mathrm{top\_k}_i$、$\mathrm{top\_p}_i$ | 第 $i$ 行请求的 top-k、top-p 参数；不分行时写 $\mathrm{top\_k}$、$\mathrm{top\_p}$ | 个 / 概率 | 旧 $k_i$、$p_i$（ch01 L579 起） |
| $\operatorname{LSE}$、$\operatorname{LSE}_i$ | Log-Sum-Exp：$\log\sum_j\exp(s_{ij})$ | 无单位 | 旧 ch09 的 $L$、$L_i$ |
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
| $N_{\mathrm{scheduled}}$、$N_{\mathrm{remaining}}$、$N_{\mathrm{chunk}}$ | 本步调度给请求的 token 数；还剩多少 token 没算；chunked prefill 本次切下的 token 数 | ch04 |
| $N_{\mathrm{budget}}$、$N_{\mathrm{budget,in}}$、$N_{\mathrm{max}}$ | 本步剩余的 token 预算；输入侧预算；`max_tokens` | ch04 |
| $N_{\mathrm{hit,local}}$、$N_{\mathrm{hit,ext}}$ | 本地、外部前缀命中的 token 数 | ch04 |
| $N_{\mathrm{ret}}$、$N_{\mathrm{look}}$、$N_{\mathrm{tail}}$ | 滑窗保留的 token 数；lookahead token 数；最后一个 block 的尾部空位 | ch05、ch06 |
| $N_{\mathrm{mm}}$ | 一个多模态 item 占的 encoder token 数 | ch04 §4.16 |
| $N_{q,r}$、$N_{\mathrm{ctx},r}$ | 请求 $r$ 本步的 query token 数；它已有的上下文 token 数，$N_{\mathrm{ctx},r}=S_r-N_{q,r}$ | ch06 |
| $N_{\mathrm{row},i}$、$N_{\mathrm{rt},i}$、$N_r$ | logits 行数；路由到 expert rank $i$ 的 token 数；PCP rank $r$ 的真实 token 数 | ch09 |
| $N_{\mathrm{in}}^{\Sigma}$、$N_{\mathrm{out}}^{\Sigma}$ | benchmark 里成功请求的输入 token 总数、输出 token 总数 | ch10 |
| $n_{\mathrm{null}}$、$n_{\mathrm{chunk}}$ | 被换成 null block 的 block 数；kernel 一次写入要切成的 chunk 数 | ch05、ch06 |
| $n_{\mathrm{step}}$、$n_{\mathrm{phase}}$、$n_{\mathrm{grp}}$ | EngineCore 的步计数；一次集合通信的阶段数；all-reduce 的分组数 | ch04、ch09 |
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

### 1.4 字节与显存：$b$ 与 $M$

| 符号 | 含义 | 单位 | 备注 |
|---|---|---|---|
| $b_{\mathrm{elem}}$ | 一个元素占的字节数（bf16 为 2） | byte/元素 | 旧 $B_{\mathrm{dtype}}$ |
| $M_{\mathrm{tok}}$ | 每 token 的 KV 字节数，$M_{\mathrm{tok}}=2LH_{\mathrm{kv}}d_hb_{\mathrm{elem}}$（单卡用 $H_{\mathrm{kv,local}}$） | byte/token | 旧 $B_{\mathrm{token}}$（ch01、ch05）；ch04 同名的 token 预算改为 $N_{\mathrm{budget}}$。ch01 L906 先给通式 $M_{\mathrm{tok}}=L(H_kd_k+H_vd_v)\,b_{\mathrm{elem}}$（K、V 的 head 数和宽度可以不同），本行是 $H_k=H_v=H_{\mathrm{kv}}$、$d_k=d_v=d_h$ 的特例；MLA 等压缩缓存不在这两个式子之内 |
| $M_{\mathrm{blk}}$、$M_{\mathrm{seq}}$ | 每 block、每序列的 KV 字节数：$M_{\mathrm{blk}}=B_{\mathrm{blk}}M_{\mathrm{tok}}$，$M_{\mathrm{seq}}=S\,M_{\mathrm{tok}}$ | byte | 旧 $B_{\mathrm{block}}$、$B_{\mathrm{sequence}}$ |
| $M_{\mathrm{total}}$、$M_{2048}$ | 全部请求的 KV 总字节数；2048 token 的 KV 字节数（ch01 算例） | byte | 旧 $B_{\mathrm{total}}$、$B_{2048}$ |
| $M_{\mathrm{KV}}$ | 可分给 KV Cache 的显存，$M_{\mathrm{KV}}=M_{\mathrm{req}}-M_{\mathrm{nonKV}}-M_{\mathrm{graph}}$ | byte | ch07 §7.9 与 ch08 L1682 是同一个量，原先分别写 `=` 与 `≈`，由 B6 核对后统一 |
| $M_{\mathrm{req}}$、$M_{\mathrm{nonKV}}$、$M_{\mathrm{graph}}$ | 请求的显存预算（`requested_memory`）；非 KV 峰值；实际扣除的 CUDA Graph 估算（`applied_cudagraph_memory_estimate`） | byte | 旧 $M_{\mathrm{requested}}$、$M_{\mathrm{graph,applied}}$、$M_{\mathrm{graph\ reserve}}$；ch07 里的 `M_util`、`M_external` 是代码样式的中间量，保持不变 |
| $M_{\mathrm{payload}}$ | 一次通信的负载字节数 | byte | 旧 $B_{\mathrm{payload}}$ |

### 1.5 并行度与排名

| 符号 | 含义 | 备注 |
|---|---|---|
| $\mathrm{TP}$、$\mathrm{PP}$、$\mathrm{DP}$、$\mathrm{EP}$、$\mathrm{PCP}$、$\mathrm{DCP}$ | 各种并行的路数（正体，不用 `p`、`P`、`N`） | 旧 ch09 的 $p$（TP 路数）、$P$（PCP 大小）、$N$（DP group 大小）；ch03、ch08 的 `P`（PP stage 数）改 $\mathrm{PP}$ |
| $W$ | world size：$W_{\mathrm{engine}}=\mathrm{TP}\times\mathrm{PP}\times\mathrm{PCP}$，$W_{\mathrm{all\_dp}}=W_{\mathrm{engine}}\times\mathrm{DP}$ | $W$ 只表示 world size，不再表示 waiting 数 |
| $g$、$\ell$、$i_{\mathrm{dp}}$ | 全局 rank、本机 rank、DP engine 序号，$g=\ell+i_{\mathrm{dp}}W$ | 旧 $d$；$\ell$ 只表示本机 rank，ch06 在线 softmax 的分母改写 $Z$ |
| $I_{\mathrm{il}}$ | DCP 的 KV interleave 大小（`dcp_kv_cache_interleave_size`） | 旧 $I$（ch09）；$I_{\mathrm{pf}}$ 是 ch04 的 `prefill_schedule_interval`（第 3 节局部字母），两者是不同的量 |

### 1.6 时间、吞吐与排队

| 符号 | 含义 | 单位 | 备注 |
|---|---|---|---|
| $\Delta t$ | 一段耗时的通式，下标说明是哪一段；ch10 §10.7.3 的请求间隔直接写 $\Delta t$ | s 或 ms | 旧 $T_{\mathrm{xxx}}$、`T`；下面几行是已登记的全部 $\Delta t_{\dots}$ |
| $\Delta t_{\mathrm{step}}$、$\Delta t_{\mathrm{sched}}$、$\Delta t_{\mathrm{prepare}}$、$\Delta t_{\mathrm{H2D}}$、$\Delta t_{\mathrm{forward}}$、$\Delta t_{\mathrm{sample}}$、$\Delta t_{\mathrm{D2H}}$、$\Delta t_{\mathrm{update}}$ | ch08 串行教学模型里一个 step 的总耗时与各段耗时 | ms | 旧 $T_{\dots}$ |
| $\Delta t_{\mathrm{layer}}$、$\Delta t_{\mathrm{compute}}$、$\Delta t_{\mathrm{collective}}$、$\Delta t_{\mathrm{imbalance}}$、$\Delta t_{\mathrm{comm}}$ | ch09 的 TP 单层耗时及其计算、集合通信、负载不均三项；一次通信的耗时 | s 或 ms | 旧 $T_{\dots}$ |
| $\Delta t_{\mathrm{EP}}$、$\Delta t_{\mathrm{dispatch}}$、$\Delta t_{\mathrm{expert}}$、$\Delta t_{\mathrm{combine}}$ | ch09 的 EP 一层耗时及 dispatch、expert 计算、combine 三段 | s 或 ms | 旧 $T_{\dots}$ |
| $\Delta t_{\mathrm{client\ queue}}$、$\Delta t_{\mathrm{queue}}$、$\Delta t_{\mathrm{prefill}}$、$\Delta t_{\mathrm{decode}}$、$\Delta t_{\mathrm{inference}}$、$\Delta t_{\mathrm{bench}}$ | ch10 一个请求的客户端排队、服务端排队、prefill、decode、inference 耗时；一次 benchmark 的时长 | s 或 ms | 旧 $T_{\dots}$；$\Delta t_{\mathrm{queue}}$ 只表示**单个请求**的排队等待（§10.11.1） |
| $\overline{\Delta t}_{\mathrm{sys}}$、$\overline{\Delta t}_{\mathrm{wait}}$ | Little 定律里请求在系统内、在队列里的**平均**停留时间 | s | 旧 $W$、$W_q$；带横线表示平均，与单个请求的 $\Delta t_{\mathrm{queue}}$ 是两个量 |
| $t_{\mathrm{xxx}}$ | 一个时间戳：$t_{\mathrm{send}}$、$t_{\mathrm{first}}$、$t_{\mathrm{last}}$、$t_j$ | s | 指标的定义靠时间戳相减：$\mathrm{TTFT}=t_{\mathrm{first}}-t_{\mathrm{send}}$ |
| $\mathrm{TTFT}$、$\mathrm{ITL}$、$\mathrm{TPOT}$、$\mathrm{E2EL}$ | 首 token 时延、token 间隔、每输出 token 耗时、端到端时延 | ms | 正体；$\mathrm{SLO}_{\mathrm{ttft}}$ 等是对应的阈值，旧 $S_{\mathrm{ttft}}$ |
| $\lambda$、$\rho$ | 请求到达率（request/s）；利用率（0 到 1），也用于 KV 使用率 $\rho_{\mathrm{kv},i}$ | | |
| $\mathrm{thr}_{\mathrm{out}}$、$\mathrm{thr}_{\mathrm{user}}$、$\mathrm{thr}_{\mathrm{GPU}}$、$\mathrm{thr}(n_{\mathrm{GPU}})$ | 吞吐：输出吞吐（token/s）、每用户（token/(s·user)）、每 GPU（token/(s·GPU)）；$\mathrm{thr}(n_{\mathrm{GPU}})$ 是 $n_{\mathrm{GPU}}$ 张卡时的吞吐 | | 旧 $Q_{\mathrm{out}}$、$Q_{\mathrm{user}}$、$Q_{\mathrm{GPU}}$、$Q_p$、$Q_1$；$Q$ 留给 Query 矩阵 |
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
| ch01、ch02 | `n`（代码样式） | `SamplingParams.n`，要采样的条数，是 API 参数，不是公式量 |
| ch04 | $\delta$、$k$、$j$、$r$、$a_{\mathrm{async}}$、$C_{\mathrm{step}}$、$C_{\mathrm{threshold}}$、$C_{\mathrm{other}}$、$I_{\mathrm{pf}}$、$\mathrm{pos}_{\mathrm{mm}}$、$\mathrm{cap\_bound}$ | §4.16.1 的 lookahead 偏移；§4.19.3 一次返回的 token 数与触发 stop 的序号；请求下标；async 在途采样步标志（0 或 1）；token 上限；`prefill_schedule_interval`；多模态 item 起点位置（`start_pos`）；`prefill_capacity_bound` 布尔量（旧 $B$）。带撇号的 $N_{\mathrm{new}}'$、$N_{\mathrm{computed}}'$、$N_{\mathrm{placeholders}}'$ 是对应量更新后的值，不单独登记 |
| ch05 | $E_i$、$i$、$j$ | 覆盖 block 与 lookahead 范围的 `extra_keys`；block 序号 |
| ch06 | $i$、$b$、$o$、$m$、$Z$、$m'$、$Z'$、$m_{\mathrm{tile}}$、$\mathrm{pos}_0$、$E_{\mathrm{max}}$、$B_{\mathrm{mgr}}$、$B_{\mathrm{ker}}$、$\kappa$、$i_{\mathrm{mgr}}$ | §6.4 地址公式里的逻辑 block 序号与物理 block ID；$o$ 是 attention 的输出向量；$m,Z$ 是在线 softmax 的当前最大值与分母（旧 $\ell$，让给 ch09 的本机 rank），带撇号是合入新片段之后的值，$m_{\mathrm{tile}}$ 是新片段的最大分数；$\mathrm{pos}_0$ 是滑窗里首个 query 的位置；$E_{\mathrm{max}}$ 是一次 staged 写入的最长元素数；manager、kernel 的 block 大小、两者之比 $\kappa=B_{\mathrm{mgr}}/B_{\mathrm{ker}}$ 与 manager block 序号 |
| ch07 | $k$、$x$ | FLy 判定里熵的 top-$k$ 大小（不是采样的 top-k）；输入 |
| ch09 | $A,A_i,X_i,Y_i,b_i$、$Y^{\mathrm{part}}_i$、$u_r,f_r,\Sigma_u,\Sigma_f$、$G_2$、$\hat T$、$\mathrm{pad}_{r,t}$、$\alpha,\beta$、$s_{ij}$、$N_{\mathrm{sch},i}$、$N_{\mathrm{drf},i}$、$N_{\mathrm{acc},i}$ | TP 线性层的权重与分片（$A$ 是权重，为避免与 world size $W$ 冲突）、偏置 $b_i$；行并行的 partial output；DP 同步的两个标志和它们的和；all-reduce 分组大小；PCP 填充后每 rank 的 token 数及其掩码；通信的延迟与带宽；某个 query 对某个 key 的打分；「v0.33.0：batch-sharded sampling 可与自适应验证同开」一节（ch09 L693）里请求 $i$ 调度时预留的 logits 行数、带的 draft token 数、实际接纳的 draft 数（$N_{\mathrm{sch},i}$ 是按调度上限预留的 logits 行数，**不等于**第 1.3 节的 $N_{\mathrm{scheduled}}$，后者是 ch04 里请求本步被调度的 token 数） |
| ch10 | $b$、$f$、$s$、$x_{\mathrm{treatment}}$、$x_{\mathrm{baseline}}$、$\Delta_{\%}$ | 到达间隔 Gamma 分布的 burstiness；Amdahl 公式里被加速部分的占比与加速倍数；对比实验的指标值与相对变化 |

## 4 改名对照表

按章列出，每行是一类写法：旧写法在哪些行、改成什么、为什么。**这一节只记录，不执行**：章节正文由 B1–B8 各批按此表改写；
改完的行不会从表里删除，B9 收尾时用它做“旧写法不得再出现”的检查。

行号规则：显示公式用 `$$` 起始行，行内公式、行内代码和 mermaid 标签用所在行。覆盖范围是 `$…$`、`$$…$$`、行内代码和 mermaid 标签；
夹在中文句子里、没有写成公式或代码的字母（例如“块大小 B”）普查不到，批次改写时要自己再读一遍。

### ch01 Transformer 推理

| 旧写法 | 新写法 | 含义 | 位置（行号） |
|---|---|---|---|
| $T$（温度） | $\tau$ | softmax 温度 | L509、L513 |
| $z_i/T$ | $z_i/\tau$ | softmax 温度 | L1519 |
| $K^{\mathsf T}$ | $K^{\top}$ | 转置（同章 L333 起另写作 `K^T`） | L324 |
| `K^T` | $K^{\top}$ | 转置 | L333 |
| $D_h$ / `D_h` / `Dh` | $d_h$（代码样式 `d_h`） | 单头宽度（Q/K） | L324、L332、L334、L337、L338、L339、L340、L398、L414、L415、L416、L916 |
| $D_k$、$D_v$ / `Dv` | $d_k$、$d_v$ | K / V 的单头宽度（仅在两者可能不同的通式里出现） | L332、L334、L416、L417、L906、L911、L912 |
| `Hq`、`Hkv` | $H_q$、$H_{\mathrm{kv}}$（形状里 `H_q`、`H_kv`） | Q / KV head 数 | L414、L415、L416、L417 |
| `Nq`、`Nk` | $N_q$、$N_k$ | 单头例子里 query / key 行数 | L331、L332、L333、L334 |
| $M$（因果 mask） | $\mathrm{Mask}$ | 因果 mask（把 $M$ 留给显存/字节数） | L324、L334 |
| 形状里的 `V`（词表） | `vocab_size` | 形状里的词表维（L136 `[T, V]`、L206 `[V]`、L1478 `[B,S,V]`） | L136、L206、L1478 |
| $V$（词表） | $\lvert\mathcal V\rvert$ | 词表大小（同章 L312 起 $V$ 是 Value 矩阵） | L206、L572、L579、L587、L589、L593、L615 |
| $k_i$、$k$（每行 top-k） | $\mathrm{top\_k}_i$、$\mathrm{top\_k}$ | 第 $i$ 行的 top-k（避免与 key $k$ 混淆） | L579、L587、L593、L594、L615 |
| $p_i$、$p$（每行 top-p） | $\mathrm{top\_p}_i$、$\mathrm{top\_p}$ | 第 $i$ 行的 top-p（$p_i$ 只留给概率；同章 L202 的 $p_i$ 是概率） | L579、L588、L590、L593、L594、L615 |
| `I`（MLP 中间宽度） | $D_{\mathrm{ff}}$ | MLP 中间宽度 | L263 |
| $B_{\mathrm{token}}$ | $M_{\mathrm{tok}}$ | 每 token 的 KV 字节数 | L903、L906、L916、L938、L948、L959 |
| $B_{\mathrm{dtype}}$ | $b_{\mathrm{elem}}$ | 每个元素的字节数 | L903、L906、L916、L927 |
| $B_{2048}$、$B_{\mathrm{total}}$ | $M_{2048}$、$M_{\mathrm{total}}$ | 2048 token / 全部请求的 KV 总字节数 | L938、L948、L956、L959 |
| `T`（温度）、`T>0` 等 | $\tau$（`τ>0` 等） | softmax 温度（同章 `[T,D]` 里的 `T` 是 token 数） | L513、L514、L516、L517、L522、L541、L553 |
| `k`、`p`（top-k、top-p 参数） | $\mathrm{top\_k}$、$\mathrm{top\_p}$ | top-k / top-p 参数 | L538、L539 |
| `P=128`、`G=32` | $N_{\mathrm{prompt}}=128$、$N_{\mathrm{out}}=32$ | 最小实验的 prompt / 输出 token 数（`G` 留给 ch10 的 goodput） | L891 |
| 上标 `T`（转置） | 上标 $\top$ | 转置（`T` 留给 token 数） | L332 |

### ch03 EngineCore

| 旧写法 | 新写法 | 含义 | 位置（行号） |
|---|---|---|---|
| `P`（PP stage 数） | $\mathrm{PP}$ | pipeline stage 数 | L624、L625 |

### ch04 Scheduler

| 旧写法 | 新写法 | 含义 | 位置（行号） |
|---|---|---|---|
| $B_{\mathrm{token}}$ | $N_{\mathrm{budget}}$ | 剩余 token 预算（原文自注「不是字节数」） | L675、L679 |
| $B_{\mathrm{input}}$ | $N_{\mathrm{budget,in}}$ | 输入侧 token 预算 | L676、L679 |
| $H_{\mathrm{local}}$、$H_{\mathrm{external}}$（及代码样式 `H_local`、`H_external`） | $N_{\mathrm{hit,local}}$、$N_{\mathrm{hit,ext}}$ | 本地 / 外部前缀命中 token 数 | L977 |
| $s$ | $\mathrm{pos}_{\mathrm{mm}}$ | 多模态 item 起点位置（`start_pos`） | L1072、L1077、L1081、L1082、L1085、L1086、L1088、L1090、L1096 |
| $L$ | $N_{\mathrm{mm}}$ | item 占的 encoder token 数（避免与层数 $L$ 冲突） | L1072、L1077、L1081、L1083、L1085 |
| $c$ | $N_{\mathrm{computed}}$ | 已计算 token 数（同章 L316 起已是 $N_{\mathrm{computed}}$） | L1073、L1077、L1081、L1083、L1085、L1086、L1087、L1088、L1090、L1096、L1201、L1209、L1219、L1220 |
| $n$、$n'$ | $N_{\mathrm{new}}$、$N_{\mathrm{new}}'$ | 本轮新调度 token 数 | L1074、L1077、L1081、L1083、L1086、L1090、L1096 |
| $n$ | $N_{\mathrm{new}}$ | 本轮调度 token 数 | L1201、L1209、L1219、L1220 |
| $T$ | $N_{\mathrm{tokens}}$ | 请求当前已知 token 总数（`num_tokens`；L983 已是 $N_{\mathrm{tokens}}$） | L1201、L1210、L1219 |
| $O$ | $N_{\mathrm{out}}$ | 已确认输出 token 数 | L1201、L1210、L1219 |
| $a$ | $a_{\mathrm{async}}$ | async 在途采样步标志（0/1） | L1201、L1210、L1220 |
| $M$ | $N_{\mathrm{max}}$ | `max_tokens` | L1201、L1211、L1219、L1221 |
| $c$、$c'$ | $N_{\mathrm{computed}}$、$N_{\mathrm{computed}}'$ | 已计算 token 数 | L1364、L1368、L1372 |
| $P$、$P'$ | $N_{\mathrm{placeholders}}$、$N_{\mathrm{placeholders}}'$ | `num_output_placeholders`（L322 已是 $N_{\mathrm{placeholders}}$） | L1364、L1368、L1369、L1372 |
| $I$ | $I_{\mathrm{pf}}$ | `prefill_schedule_interval` | L1497、L1498、L1500、L1508、L1509、L1513、L1518、L1522、L1524、L1525 |
| $N$ | $n_{\mathrm{step}}$ | 步计数（无单位整数） | L1513、L1518、L1522、L1524、L1525 |
| $B$ | $\mathrm{cap\_bound}$ | `prefill_capacity_bound` 布尔量（$B$ 只留给批大小） | L1513、L1519、L1523、L1524、L1528、L1543 |
| `H_local`、`H_external` | $N_{\mathrm{hit,local}}$、$N_{\mathrm{hit,ext}}$ | 前缀命中 token 数 | L974、L975 |
| `% I` | `% I_pf` | `prefill_schedule_interval` | L1508、L1509 |
| 图中 `c`、`n`、`s`、`T`、`O`、`a` | `N_computed`、`N_new`、`pos_mm`、`N_tokens`、`N_out`、`a_async` | 判定节点里的符号（纯文本） | L1102、L1103、L1228、L1230 |
| 图中 `N`、`I` | `n_step`、`I_pf` | prefill 节拍图里的符号（纯文本） | L1532 |

### ch05 KV Cache

| 旧写法 | 新写法 | 含义 | 位置（行号） |
|---|---|---|---|
| $B_{\mathrm{token}}$ | $M_{\mathrm{tok}}$ | 每 token 的 KV 字节数 | L168、L182、L186、L232 |
| $B_{\mathrm{sequence}}$ | $M_{\mathrm{seq}}$ | 每序列的 KV 字节数 | L182、L247 |
| $B_{\mathrm{block}}$ | $M_{\mathrm{blk}}$ | 每 block 的 KV 字节数 | L186、L241 |
| $B_{\mathrm{dtype}}$ | $b_{\mathrm{elem}}$ | 每个元素的字节数 | L168、L178 |
| $D_{\mathrm{head}}$ | $d_h$ | 单头宽度 | L168、L177 |
| `T`（block 大小） | $B_{\mathrm{blk}}$ | block 大小（token 数） | L180、L186 |
| $T$、$W_{\mathrm{tail}}$ | $B_{\mathrm{blk}}$、$N_{\mathrm{tail}}$ | block 大小；最后一个 block 的尾部空位 token 数 | L365 |
| $T$ | $B_{\mathrm{blk}}$ | block 大小 | L1644、L1653 |
| $R$ | $N_{\mathrm{ret}}$ | 滑窗保留 token 数（`num_retained_tokens`） | L1644、L1652 |
| $p$ | $\mathrm{pos}$ | token 位置 | L1644、L1651 |
| $H_i$ | $\eta_i$ | 第 $i$ 个 block 的链式 hash（图里写作 `hash_i`） | L1102、L1224、L1230 |
| $\mathrm{tokens}_i^{+L}$、`L` | $\mathrm{tokens}_i^{+N_{\mathrm{look}}}$ | lookahead token 数（避免与层数 $L$ 冲突） | L1224、L1232、L1234、L1235、L1239、L1240 |
| 图中 `tokens 0 到 B-1` 等 | `tokens 0 到 B_blk-1` 等 | mermaid 标签用纯文本 | L1110、L1112、L1114 |
| `L`（lookahead） | $N_{\mathrm{look}}$ | lookahead token 数 | L1241、L1242、L1264、L1536 |
| `R`、`T`、`p` | $N_{\mathrm{ret}}$、$B_{\mathrm{blk}}$、$\mathrm{pos}$ | 滑窗 null block 的算例（`n_null` 本身不改名） | L1651、L1652、L1653、L1657、L1659 |
| 图中 `L = 0`、`L = num_prefill_lookahead_tokens` | `N_look = 0`、`N_look = num_prefill_lookahead_tokens` | lookahead（纯文本） | L1248、L1254 |
| 图中 `p = …`、`R = …` | `pos = …`、`N_ret = …` | 滑窗算例图（纯文本） | L1668、L1669 |

### ch06 PagedAttention

| 旧写法 | 新写法 | 含义 | 位置（行号） |
|---|---|---|---|
| $B$ | $B_{\mathrm{blk}}$ | block 大小 | L233、L236、L240、L248、L499、L512、L515、L1445 |
| $p$ | $\mathrm{pos}$ | token 位置 | L232、L236、L240、L277 |
| $o$（block 内偏移） | $\mathrm{off}$ | block 内偏移（$o$ 留给 attention 输出向量） | L240、L248、L255、L277 |
| $p_0$ | $\mathrm{pos}_0$ | 窗口里首个 query 的位置 | L499、L503、L508、L515 |
| $R$ | $N_{\mathrm{ret}}$ | 滑窗保留 token 数 | L499、L503、L509、L515 |
| $L_{\max}$ | $E_{\mathrm{max}}$ | 一次 staged 写入的最长元素数（避免与层数 $L$ 冲突） | L552、L560、L565 |
| $q_r$ | $\mathrm{qsl}_r$ | `query_start_loc[r]`（$q$ 留给 query 向量） | L706、L712 |
| $L^Q_r$ | $N_{q,r}$ | 请求 $r$ 本步的 query token 数 | L712、L730 |
| $L^{\mathrm{seq}}_r$、$L^{\mathrm{context}}_r$ | $S_r$、$N_{\mathrm{ctx},r}$ | 请求 $r$ 的可见序列长度；已有上下文 token 数 | L730 |
| $\sqrt d$ | $\sqrt{d_h}$ | 单头宽度 | L1431 |
| $p_j$（attention 权重） | $a_j$ | attention 权重（$p_i$ 只留给概率） | L1435、L1439 |
| `B=4` | $B_{\mathrm{blk}}=4$ | block 大小 | L275 |
| $\ell$、$\ell'$ | $Z$、$Z'$ | 在线 softmax 的归一化分母（$\ell$ 留给 ch09 的本机 rank） | L1527、L1534、L1538、L1541 |
| `A`、`K`、`r=A/K`、`m` | $B_{\mathrm{mgr}}$、$B_{\mathrm{ker}}$、$\kappa=B_{\mathrm{mgr}}/B_{\mathrm{ker}}$、$i_{\mathrm{mgr}}$ | manager / kernel block 大小与换算（`A/K/r/m` 与矩阵 $A,K$、请求下标 $r$、在线 softmax 的 $m$ 冲突） | L316、L317、L318、L348、L365 |
| 图中 `b times B` | `b times B_blk` | 地址公式图（纯文本） | L264、L265 |
| 图中 `n_null = floor(max(p0-R, 0) / B)` | `n_null = floor(max(pos_0-N_ret, 0) / B_blk)` | null block 图（纯文本） | L522 |

### ch07 模型执行

| 旧写法 | 新写法 | 含义 | 位置（行号） |
|---|---|---|---|
| $M_{\mathrm{requested}}$ | $M_{\mathrm{req}}$ | `requested_memory` | L611 |
| $M_{\mathrm{graph},\mathrm{applied}}$ | $M_{\mathrm{graph}}$ | `applied_cudagraph_memory_estimate` | L611、L621 |
| $T$（温度） | $\tau$ | softmax 温度 | L1100 |
| $H(x)$、$H$ | $\mathcal H(x)$、$\mathcal H$ | top-k 熵（$H$ 留给 head 数） | L1353、L1357、L1359、L1361 |
| `T>0`、`T=0`、`T=1` | `τ>0`、`τ=0`、`τ=1` | softmax 温度 | L1104、L1105 |

### ch08 compile / CUDA Graph

| 旧写法 | 新写法 | 含义 | 位置（行号） |
|---|---|---|---|
| $T_{\mathrm{step}}$、$T_{\mathrm{sched}}$ 等 | $\Delta t_{\mathrm{step}}$、$\Delta t_{\mathrm{sched}}$ 等 | 耗时 | L128、L131 |
| $M_{\mathrm{available\ KV}}$、$M_{\mathrm{graph\ reserve}}$ | $M_{\mathrm{KV}}$、$M_{\mathrm{graph}}$ | 可给 KV 的显存、graph 预留 | L1682 |
| $M_{\mathrm{requested}}$ | $M_{\mathrm{req}}$ | `requested_memory` | L1682 |
| `P`（PP stage 数） | $\mathrm{PP}$ | pipeline stage 数 | L287 |

### ch09 分布式

| 旧写法 | 新写法 | 含义 | 位置（行号） |
|---|---|---|---|
| $d$（DP 序号） | $i_{\mathrm{dp}}$ | DP engine 序号 | L171、L178、L180 |
| $A_{p-1}$、$X_{p-1}$、`p` | $A_{\mathrm{TP}-1}$、$X_{\mathrm{TP}-1}$ | TP 路数（$p$ 只留给概率/位置的旧用法） | L432、L443、L447、L469 |
| $P_i$、`P_i` | $Y^{\mathrm{part}}_i$ | 行并行下第 $i$ 个分片的 partial output | L478、L484、L488 |
| $T_{\mathrm{layer}}(p)$ 等 | $\Delta t_{\mathrm{layer}}(\mathrm{TP})$ 等 | TP 路数与耗时 | L553、L557 |
| $n_i$ | $N_{\mathrm{row},i}$ | logits 行数 | L693、L700、L701、L702 |
| $s_i$ | $N_{\mathrm{sch},i}$ | 本步调度的 token 数 | L693、L699、L701 |
| $d_i$ | $N_{\mathrm{drf},i}$ | draft token 数（避免与 $d$ 冲突） | L693、L699、L700、L701 |
| $a_i$ | $N_{\mathrm{acc},i}$ | 被接受的 draft token 数 | L693、L700、L701 |
| $W_i$（waiting） | $n_{\mathrm{wait},i}$ | engine $i$ 的 waiting 请求数（$W$ 只指 world size） | L820、L826 |
| $R_i$ | $n_{\mathrm{run},i}$ | engine $i$ 的 running 请求数 | L820 |
| $U_i$ | $\rho_{\mathrm{kv},i}$ | KV 使用率（0–1） | L826 |
| $C$ | $n_{\mathrm{cli}}$ | client 数 | L820、L830 |
| $I_i$ | $n_{\mathrm{fly},i}$ | client 观察到的 engine $i$ in-flight 数 | L820、L830 |
| $N$（DP group 大小） | $\mathrm{DP}$ | DP group 大小（`dp_group.size()`） | L923、L927、L932、L934、L942、L949、L957、L972 |
| $S_0$ | $\Sigma_u$ | `has_unfinished` 之和 | L923、L927、L935、L954 |
| $S_1$ | $\Sigma_f$ | 暂停标志之和 | L923、L927、L932、L935 |
| $p_r$ | $f_r$ | rank $r$ 是否有待处理的暂停请求（0/1） | L923、L935 |
| $B$（块大小） | $G_2$ | all-reduce 的 2 的幂分组大小 | L942、L945、L946、L949、L953、L957 |
| $M=N/B$ | $n_{\mathrm{grp}}=\mathrm{DP}/G_2$ | 分组数 | L942、L949、L957 |
| $P$ | $\mathrm{PCP}$ | PCP 大小 | L1261、L1268、L1275、L1277 |
| $m_{r,t}$ | $\mathrm{pad}_{r,t}$ | padding 掩码（$m$ 留给在线 softmax 的最大值） | L1268、L1276 |
| $n_r$ | $N_r$ | rank $r$ 的真实 token 数 | L1268、L1275、L1277 |
| $I$ | $I_{\mathrm{il}}$ | `dcp_kv_cache_interleave_size` | L1308、L1312 |
| $p$ | $\mathrm{pos}$ | token 位置 | L1308、L1312 |
| $L$、$L_i$、`L_i` | $\mathrm{LSE}$、$\mathrm{LSE}_i$ | Log-Sum-Exp（$L$ 只指层数） | L1334、L1336、L1342 |
| $n_i$ | $N_{\mathrm{rt},i}$ | 路由到 expert rank $i$ 的 token 数 | L1058、L1060 |
| $T_{\mathrm{EP}}$、$T_{\mathrm{dispatch}}$ 等 | $\Delta t_{\mathrm{EP}}$、$\Delta t_{\mathrm{dispatch}}$ 等 | 耗时 | L1060 |
| $T_{\mathrm{comm}}$ | $\Delta t_{\mathrm{comm}}$ | 耗时 | L1728 |
| $B_{\mathrm{payload}}$ | $M_{\mathrm{payload}}$ | 通信负载字节数 | L1728、L1733 |
| `Q_1`、$Q_p$、`Q_p`、$p$（卡数） | $\mathrm{thr}(1)$、$\mathrm{thr}(n_{\mathrm{GPU}})$、$n_{\mathrm{GPU}}$ | $n_{\mathrm{GPU}}$ 卡时的吞吐与卡数（$Q$ 留给 Query 矩阵） | L1739、L1741、L1745 |
| `Dout/p`、`Din/p` | `D_out/TP`、`D_in/TP` | TP 路数 | L444、L488 |
| 图中 `u, p`、`B = N`、`M` | `u, f`、`G_2`（`DP` 的最大 2 的幂因子）、`n_grp` | DP all-reduce 图（纯文本） | L961、L965、L966 |

### ch10 实验方法

| 旧写法 | 新写法 | 含义 | 位置（行号） |
|---|---|---|---|
| `T`、$T$（benchmark 时长） | $\Delta t_{\mathrm{bench}}$ | benchmark 时长 | L283、L286、L290、L294、L307 |
| `R` | $n_{\mathrm{req}}$ | 成功请求数（请求数是计数，用 $n_{\dots}$） | L283、L286 |
| `I` | $N_{\mathrm{in}}^{\Sigma}$ | 成功请求的输入 token 总数 | L284、L294 |
| `O` | $N_{\mathrm{out}}^{\Sigma}$ | 成功请求的输出 token 总数 | L284、L290、L294 |
| $S_{\mathrm{ttft}}$、$S_{\mathrm{tpot}}$、$S_{\mathrm{e2el}}$ | $\mathrm{SLO}_{\mathrm{ttft}}$、$\mathrm{SLO}_{\mathrm{tpot}}$、$\mathrm{SLO}_{\mathrm{e2el}}$ | SLO 阈值 | L303、L304 |
| $R_{\mathrm{SLO}}$ | $n_{\mathrm{SLO}}$ | 达标请求数 | L305、L307 |
| $N$ | $N_{\mathrm{out}}$ | 单个请求的输出 token 数 | L150、L170、L183、L185 |
| $T_{\mathrm{client\ queue}}$ | $\Delta t_{\mathrm{client\ queue}}$ | 客户端排队耗时 | L152 |
| $T_{\mathrm{queue}}$、$T_{\mathrm{prefill}}$、$T_{\mathrm{decode}}$、$T_{\mathrm{inference}}$ | $\Delta t_{\mathrm{queue}}$、$\Delta t_{\mathrm{prefill}}$、$\Delta t_{\mathrm{decode}}$、$\Delta t_{\mathrm{inference}}$ | 单个请求的四段耗时（$\Delta t_{\mathrm{queue}}$ 只指单个请求的排队等待） | L729、L733、L737、L741 |
| $L$、$L_q$ | $n_{\mathrm{sys}}$、$n_{\mathrm{queue}}$ | 系统内 / 队列内平均请求数（$L$ 只指层数） | L909、L911 |
| $W$、$W_q$ | $\overline{\Delta t}_{\mathrm{sys}}$、$\overline{\Delta t}_{\mathrm{wait}}$ | 系统内 / 队列内的**平均**停留时间（$W$ 只指 world size；带横线，与 L729 单个请求的 $\Delta t_{\mathrm{queue}}$ 区分） | L909、L910、L911 |
| $S_{\mathrm{total}}$ | $\mathrm{speedup}_{\mathrm{total}}$ | 端到端加速比（与 ch09 L1741 的 $\mathrm{speedup}$ 同族） | L1255 |
| `N>1` | $N_{\mathrm{out}}>1$ | 输出 token 数 | L168、L183、L185 |
| $Q_{\mathrm{out}}$、$Q_{\mathrm{user}}$、$Q_{\mathrm{GPU}}$ | $\mathrm{thr}_{\mathrm{out}}$、$\mathrm{thr}_{\mathrm{user}}$、$\mathrm{thr}_{\mathrm{GPU}}$ | 吞吐（$Q$ 留给 Query 矩阵） | L1472、L1475、L1479、L1483 |
| $N_{\mathrm{users}}$、$N_{\mathrm{GPU}}$ | $n_{\mathrm{users}}$、$n_{\mathrm{GPU}}$ | 用户数、GPU 数（计数，用 $n_{\dots}$；$N_{\dots}$ 的单位是 token） | L1472、L1473、L1475、L1479 |

## 5 写法规则

1. 数学量写成 `$…$`；代码标识符、路径、参数名和程序输出写成代码样式，不把它们冒充数学。
2. 张量形状写成代码样式，维度名取自第 1 节：`[B,S,H_q,d_h]`、`[T,H_kv,d_h]`、`[T,D_in]`；一个形状里每个维度名只表示一个量。
3. 有量纲的量写单位并正体：`128\,\mathrm{KiB/token}`；乘法链里把单位一起带上做单位检查。
4. 描述性下标、并行度、指标名用 `\mathrm{…}`，函数名用 `\operatorname{…}`；舍入写 `\approx`；多步推导用 `aligned`。
5. mermaid 的标签不能用 LaTeX 或 HTML（`epub/mermaid-config.json` 里 `htmlLabels` 为 `false`），符号写成纯文本：
   `d_h`、`B_blk`、`M_tok`、`hash_i`、`tau`。
6. 公式前后的文字：首次出现时必须说明每个符号、shape、单位和这个公式回答的问题（`AGENTS.md` §8）；用第 6 节的模板写。
7. 新增一个全局符号要改本文件的第 1 节，只有 F0 和 B9 可以改；批次里发现缺项，写进 PR 正文，由 B9 汇总。

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

**核对：**与源码、`examples/` 或测试的哪一处一致；能用脚本算的，命令和输出写进 PR 正文。
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

**核对：**与 ch01 L948、ch05 L232 的算例一致。

## 7 配图规则

目标是复杂的地方一定有图，但不多。

- **必配的 10 处**（tracker 清单里的 R 项，各批必须补）：
  ch01 单头 attention 数据流（§「Q、K、V 是什么」）、ch01 GQA 的 Q head 与 KV head 对应（§「Multi-Head 与 GQA」）；
  ch02 §2.15 五种队列的拓扑；ch03 §「v0.33.0：IPC 与共享内存不是同一个开关」；
  ch04 §4.19.3 speculative rejection 回滚的时间线；
  ch05 §5.5.3 多个 cache group 共享 block ID pool、§5.15.5 checkpoint 位置的三个分支；
  ch09 §9.2.1 三种 rank 与 GPU 编号、§9.9.5 partial attention 与 LSE 合并；
  ch10 §10.11.1 请求内部时间拆分。
- **review 提出的要求**：review 在某一批里要求补图时，该批照做，并在 PR 正文里写明依据。
- **可选的位置**（tracker 里的 O 项）只在**确实复杂**时才做：同时满足“涉及三个以上互相依赖的对象或状态，或一段文字在讲分支、顺序、坐标映射”
  与“读者没有图就很难复述主路径”。**每个批次 PR 的正文必须列出做了哪些可选位置及原因，没做的也写一句原因。**
  候选清单：ch01 RoPE 成对旋转、SamplingParams 规范化的判定顺序；ch02 Renderer 的步骤、ch03 §3.6.2 KV 初始化的四步；
  ch04 §4.17.4 `encoder_input_updates` payload；ch05 §5.2.3 通用 page 公式的组成、§5.9 null block 与 block table/free queue；
  ch06 §6.5.1 packed layout 的 stride 映射；ch07 §7.3.2 external launcher 的对应关系、ch08 §8.14.6 piecewise 的两种实现来源；
  ch09 §9.3.1 世界规模的因子嵌套、§9.7.3 负载分数选择流程、§9.8.2 expert placement、§9.9.3 PCP 配置校验；
  ch10 §10.11.5 服务端耗时与客户端 TTFT 对账、§10.20.2 `bench sweep` 流程。
- **每张图**（`AGENTS.md` §9）：只表达一个主要问题；节点能对应真实的模块、类或对象；关键边在 PR 正文里写出源码依据；
  标签用纯文本（第 5 节）；图后紧跟手写的读图方法引用块（格式同各章已有的图），不用通用文案；在本地用 mermaid-cli 渲染并看过 PNG（仓库不提交渲染产物）。

## 8 B1–B8 的优先清单：没有任何讲解线索的 27 组行间公式

“没有线索”的判据是：公式前 5 行到后 12 行内，没有“其中/符号/令/设/表示”之类的符号定义，也没有“白话/读法/直觉”之类的通俗讲解（启发式，
`scripts` 里没有对应检查；每批仍要逐个重读）。这 27 组在各批里最先处理，并按第 6 节补全。行号对应 main `14708f03`。

| 批 | 章 | 行号 | 公式 |
|---|---|---|---|
| B1 | ch01 | L312 | $Q=XW_Q$、$K=XW_K$、$V=XW_V$ 投影 |
| B1 | ch01 | L379 | 单头例子里的 softmax 权重向量 $a$ |
| B1 | ch01 | L385 | 用权重对 $V$ 加权求和 |
| B1 | ch01 | L471 | RoPE 的二维旋转矩阵 |
| B1 | ch01 | L938 | KV 字节数算例（$M_{\mathrm{tok}}$、$M_{2048}$，旧 $B_{\mathrm{token}}$） |
| B3 | ch04 | L316 | $N_{\mathrm{new}}=N_{\mathrm{target}}-N_{\mathrm{computed}}$ |
| B3 | ch04 | L322 | 含 speculative token 与 placeholder 的 $N_{\mathrm{new}}$ |
| B3 | ch04 | L977 | 前缀命中后的 $N_{\mathrm{computed}}$ |
| B3 | ch04 | L983 | 前缀命中后的剩余 $N_{\mathrm{new}}$ |
| B4 | ch05 | L232 | $M_{\mathrm{tok}}$ 算例（128 KiB/token） |
| B4 | ch05 | L241 | $M_{\mathrm{blk}}$ 算例（8 MiB） |
| B4 | ch05 | L1102 | 链式 hash $\eta_i$ |
| B5 | ch06 | L706 | query 区间 $[\mathrm{qsl}_r,\mathrm{qsl}_{r+1})$ |
| B5 | ch06 | L712 | 请求 $r$ 的 query 长度 $N_{q,r}$ |
| B5 | ch06 | L1431 | 分页不改数学：$s_j$、$a_j$、$o$（一组，L1431–L1439） |
| B5 | ch06 | L1445 | 分页取 K/V 的地址式 |
| B6 | ch08 | L297 | $4\times(2+5)=28\ \mathrm{ms}$ |
| B6 | ch08 | L408 | 新增 placeholder 数 |
| B7a | ch09 | L432 | 列并行的切分 |
| B7a | ch09 | L469 | 行并行的切分 |
| B7a | ch09 | L478 | 行并行的 partial output $Y^{\mathrm{part}}_i$ |
| B7a | ch09 | L484 | partial output 求和 |
| B7a | ch09 | L820 | engine 负载分数 |
| B7b | ch09 | L1336 | LSE 合并 |
| B7b | ch09 | L1741 | 扩展效率 |
| B8 | ch10 | L170 | $\mathrm{TPOT}$ |
| B8 | ch10 | L729 | queue / prefill / decode / inference 四段耗时 |

其余有部分线索（有符号定义但没有“怎么算出来”）的公式组，同样要按模板补第三段；它们排在上面 27 组之后。

## 9 共享文件与改动权限

- 本文件（`glossary/notation.md`）：只有 F0 与 B9 可以改。批次里发现缺项，记在 PR 正文，B9 汇总。
- `glossary/terms.md`、`book/README.md`、`book/reading-guide.md`、`meta/roadmap.md`：F0 与 B9 改；其余批次不动。
- 每批只改自己的章节文件；`meta/chapter-source-map.json`、`meta/chapter-source-map.md` 一般不用改，万一改，合并时两边都保留。
- 章节保持 `draft`、`runtime_verified: false`；源码引用保持 pin `05caae03c20f69436ff7ec23ded339ea104f1a01`。
