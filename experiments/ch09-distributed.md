# 第09章分布式实验记录

2026-09-08 全书复核补记：本地 CPU 重跑的版本、环境、命令与结果统一记录在
[全书检查报告](../meta/full-book-review-2026-09-08.md)。下文历史输出保留原日期；GPU
实验表格是待执行模板，空白不代表零值或测试通过。

> 本文件用于真实 NVIDIA GPU、NCCL、Ray 或多机环境。不要用教学模拟器的结果替代真实
> 运行记录；没有执行的项目填写“未执行”和原因。

## 1. 实验目标

- 待验证假设：
- 对照组：
- 实验组：
- 正确性门槛：
- 性能指标与 SLO：
- 本实验不能推出的结论：

## 2. 固定信息

| 项目 | 记录 |
|---|---|
| 日期 | |
| vLLM commit | `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b` |
| 模型与 revision | |
| tokenizer revision | |
| 节点数 / 每节点 GPU | |
| GPU 型号与显存 | |
| GPU 拓扑 | |
| Driver / CUDA / PyTorch | |
| NCCL 版本 | |
| 网络接口与 transport | |
| Executor backend | |
| 环境变量 | |
| 完整启动命令 | |

## 3. 解析后的并行配置

| 配置 | 值 |
|---|---:|
| TP | |
| PP | |
| DP | |
| PCP | |
| DCP | |
| EP enabled / size | |
| 每 DP engine worker world size | |
| 跨 DP 总 worker 数 | |
| local world size | |

## 4. rank 与 group 表

| global rank | node | local rank | CUDA device | DP | PP | PCP | TP | TP group | PP group | DP group | EP group | DCP group |
|---:|---|---:|---|---:|---:|---:|---:|---|---|---|---|---|
| | | | | | | | | | | | | |

核对项：

- [ ] 每个 global rank 唯一。
- [ ] local rank 与物理 GPU 绑定一致。
- [ ] 每个 group 成员数符合配置。
- [ ] 所有 ranks 对 world size、master address 与 group 顺序达成一致。

## 5. workload

| 项目 | 记录 |
|---|---|
| 请求数 | |
| 输入长度分布 | |
| 输出长度分布 | |
| request rate / concurrency | |
| sampling 参数与 seed | |
| warmup | |
| 重复次数 | |
| prefix / multimodal / LoRA | |

## 6. 正确性

| 检查 | 单卡基线 | 多卡结果 | 容差 / 判定 | 结论 |
|---|---|---|---|---|
| logits 或 top-k | | | | |
| greedy token 序列 | | | | |
| 请求完成数 | | | | |
| 输出长度 | | | | |
| abort / timeout | | | | |

## 7. TP 对照

| TP | batch / token shape | 每 rank 权重 GiB | KV GiB | prefill ms | decode ITL ms | collective ms | output tok/s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | | |
| 2 | | | | | | | |

记录 Column/Row Parallel 关键 tensor 的全局 shape、本地 shape、collective 与 group。

## 8. PP 对照

| PP | stage layer ranges | batch queue | activation shape | send/recv ms | bubble % | peak GiB | output tok/s |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | | | | | | | |
| 2 | | | | | | | |

检查异步 send handle 完成前 buffer 是否保持不变，并保存 timeline 截图路径。

## 9. DP 时间序列

| 时间 | engine | waiting | running | KV usage | local inflight | score | 新请求归属 |
|---|---:|---:|---:|---:|---:|---:|---|
| | | | | | | | |

汇总：

| DP | request/s | input tok/s | output tok/s | p50 TTFT | p99 TTFT | p99 ITL | error % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | | |
| 2 | | | | | | | |

## 10. EP token 分布

| layer | EP rank | local experts | routed tokens | max expert tokens | dispatch ms | expert ms | combine ms |
|---|---:|---|---:|---:|---:|---:|---:|
| | | | | | | | |

| placement | all-to-all backend | p50 tok/s | p99 latency | max/min rank tokens | 结论 |
|---|---|---:|---:|---:|---|
| linear | | | | | |
| round_robin | | | | | |

## 11. PCP / DCP

| 模式 | 配置 | token owner / chunks | attention backend | graph mode | 通信 ms | peak GiB | 正确性 |
|---|---|---|---|---|---:|---:|---|
| baseline | | | | | | | |
| PCP | | | | | | | |
| DCP | | | | | | | |

DCP 必须额外记录 partial output、LSE 和合并路径，确认不是简单平均。

## 12. 故障注入

| 注入故障 | 预期失败层 | 首个错误 / 最后日志 | 超时 | 是否定位到 rank | 清理是否完整 |
|---|---|---|---:|---|---|
| world size 不一致 | rendezvous | | | | |
| GPU 绑定重复 | device init / OOM | | | | |
| collective shape 不一致 | runtime collective | | | | |
| 终止一个 worker | executor / monitor | | | | |

## 13. 结论

- 假设是否成立：
- 收益或退化的主要归因：
- 关键证据：
- 正确性结果：
- 适用的 workload 和硬件范围：
- 尚未解释的异常：
- 后续实验：
