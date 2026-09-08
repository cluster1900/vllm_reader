# 第10章性能实验与优化报告

2026-09-08 全书复核补记：本地 CPU 重跑的版本、环境、命令与结果统一记录在
[全书检查报告](../meta/full-book-review-2026-09-08.md)。下文历史输出保留原日期；GPU
实验表格是待执行模板，空白不代表零值或测试通过。

> 这是可复现实验记录，不是结论先行的调参日志。每个 treatment 只改变已声明的因素，
> 所有结果同时经过性能与正确性门槛。

## 1. 问题与假设

- 用户场景：
- 当前问题：
- 主指标：
- SLO：
- 可证伪假设：
- 预期内部信号：
- 可能代价：
- 不能由本实验推出的结论：

## 2. 固定环境

| 项目 | 记录 |
|---|---|
| 日期 | |
| vLLM commit / dirty diff | `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b` / |
| 模型 revision | |
| tokenizer revision | |
| quantization / KV dtype | |
| GPU / 数量 / 拓扑 | |
| CPU / NUMA / RAM | |
| Driver / CUDA / PyTorch | |
| 容器或依赖 lock | |
| power / clock policy | |
| 网络与客户端位置 | |
| 环境变量 | |
| 服务启动命令 | |
| benchmark 命令 | |

## 3. workload

| 项目 | 记录 |
|---|---|
| dataset / revision | |
| seed | |
| num prompts | |
| input min/p50/p90/p99/max | |
| output min/p50/p90/p99/max | |
| request classes | |
| request rate | |
| burstiness / ramp-up | |
| max concurrency | |
| streaming / ignore EOS | |
| sampling 参数 | |
| prefix 重复结构 | |
| timeout / retry | |

## 4. 状态与计时边界

| 项目 | 记录 |
|---|---|
| cold / steady-state | |
| warmup 请求及 shape | |
| prefix cache 初始状态 | |
| compile cache 状态 | |
| CUDA Graph capture 状态 | |
| 是否包含 detokenize | |
| TTFT 起止点 | HTTP send -> first valid streamed chunk |
| E2EL 起止点 | HTTP send -> endpoint-specific final measured event（记录是否含 usage-only） |
| client queue 是否单列 | |
| profiler 是否开启 | |

## 5. Baseline 重复结果

| run | completed | failed | req/s | goodput | input tok/s | output tok/s | p50 TTFT | p99 TTFT | p99 TPOT | p99 ITL | p99 E2EL | peak GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | | | | | | | |
| 2 | | | | | | | | | | | | |
| 3 | | | | | | | | | | | | |

- Throughput mean / std / CV：
- Tail latency mean / std / CV：
- 原始结果目录：

## 6. 内部观测

| run | waiting p50/p99 | running p50/p99 | KV usage max | preemptions | computed prompt tokens | cache-hit tokens | graph fallback | GPU busy | communication |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | | | | |

请求时间拆分（必须按同一请求对齐再求差，不能用 E2EL p99 减各阶段 p99 得到网络耗时）：

| request class | client queue p99 | server queue p99 | prefill p99 | decode p99 | network/front-end residual |
|---|---:|---:|---:|---:|---:|
| | | | | | |

## 7. Profiler 证据

| 项目 | Baseline | Treatment |
|---|---|---|
| profiler 与配置 | | |
| trace 文件 | | |
| 关键 CPU range | | |
| 关键 GPU kernel | | |
| kernel launch gap | | |
| collective fraction | | |
| graph replay / eager | | |
| 最慢 rank / stage | | |

注意：profiler run 只用于归因，性能数值来自未开启 profiler 的 benchmark run。

## 8. Treatment

- 唯一改动：
- 改动机制：
- 其他配置为何保持不变：
- correctness gate：

| run | completed | failed | req/s | goodput | input tok/s | output tok/s | p50 TTFT | p99 TTFT | p99 TPOT | p99 ITL | p99 E2EL | peak GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | | | | | | | |
| 2 | | | | | | | | | | | | |
| 3 | | | | | | | | | | | | |

## 9. A/B 比较

| 指标 | Baseline mean | Treatment mean | change % | 改善方向 | 超过噪声吗 |
|---|---:|---:|---:|---|---|
| request throughput | | | | 越高越好 | |
| goodput | | | | 越高越好 | |
| output throughput | | | | 越高越好 | |
| p99 TTFT | | | | 越低越好 | |
| p99 TPOT | | | | 越低越好 | |
| p99 E2EL | | | | 越低越好 | |
| peak memory | | | | 按目标判定 | |
| error rate | | | | 越低越好 | |

## 10. 正确性与稳定性

| 检查 | Baseline | Treatment | 容差 / 门槛 | 结果 |
|---|---|---|---|---|
| greedy token IDs | | | | |
| logits / logprobs | | | | |
| output lengths | | | | |
| finish reasons | | | | |
| failed / timeout | | | | |
| NaN / corrupted | | | | |
| abort 后 KV 释放 | | | | |
| soak 内存增长 | | | | |
| 过载恢复 | | | | |

## 11. 饱和与 Pareto 扫描

| config | offered RPS / concurrency | req/s | goodput | output tok/s | p99 TTFT | p99 TPOT | tokens/s/GPU | peak GiB | dominated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| | | | | | | | | | |

- 饱和拐点：
- SLO 内最大负载：
- Pareto frontier 配置：
- 最终选择及业务权重：

## 12. 结论

- 假设是否成立：
- 性能变化：
- 正确性是否通过：
- 证据链：
- Amdahl 一致性检查：
- 适用范围：
- 未解释异常：
- 是否建议合入/部署：
- 后续实验：
