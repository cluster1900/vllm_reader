# 第 04 章实验记录：Token 级调度与 Continuous Batching

## 目的

在没有 CUDA、PyTorch 和真实模型权重的环境中，验证第 04 章依赖的控制流不变量：

1. 新请求可以在旧请求结束前加入后续 step；
2. running 请求先消耗预算，剩余预算可接纳 waiting；
3. 长 prefill 可以切片并与普通 decode 共存；
4. scheduled token、input token 和 sequence slot 分别受限；
5. 关闭 chunked prefill 后不会绕过放不下的队首长请求；
6. KV 压力会按 FCFS 或 priority 选择 victim；
7. 抢占释放 KV，并把 computed 重置为 0 后重新计算。

## 运行命令

```bash
python3 examples/ch04_continuous_batching.py
python3 -m unittest tests/test_ch04_continuous_batching.py
```

## 演示工作负载

```text
A: arrival=0, prompt=12, max_new=3
B: arrival=1, prompt=3,  max_new=2
C: arrival=2, prompt=2,  max_new=2

scheduled token budget=8
long prefill threshold=6
max running sequences=3
```

预期 trace：

```text
step=00 work=[A:p6]
step=01 work=[A:p6, B:p2]
step=02 work=[A:d1, B:p1, C:p2]
step=03 work=[A:d1, B:d1, C:d1]
```

这组结果证明 B、C 不需要等待 A 完成；同一 step 还可以同时包含 decode 与 partial
prefill。

## 实验一：Token Budget

把 `max_num_scheduled_tokens` 分别设为 `4、8、16`，保持 workload 不变。记录：

| 指标 | 解释 |
|---|---|
| total steps | 整组请求排空所需轮数 |
| first scheduled step | 简化 TTFT 前置指标 |
| tokens per step | 预算利用率 |
| prefill chunks | 长 prompt 被切分次数 |

预期不是“预算越大所有指标都越好”。教学模型没有真实执行时间，只能显示轮数和队列
效果；真实 GPU 上更大 step 可能拉长 TPOT。

## 实验二：Chunked Prefill

构造队列：

```text
long:  prompt=10
short: prompt=2
budget=8
```

关闭 chunked prefill 时，long 整体放不进 budget，当前 step 停止准入，short 不能越过。
开启后，long 可先获得 8 个 token 或受 threshold 截断的 chunk。

## 实验三：KV Pressure

使用两个 prompt=4、max_new=3 的请求，KV 容量设为 9，并设置
`reserve_full_prompt=False`。两个请求可先同时进入 running，随后 decode 扩容超过容量，
触发抢占。

在 priority 策略下设置：

```text
low.priority=5
high.priority=0
```

预期 low 被抢占，状态变为 PREEMPTED：

```text
low.num_computed_tokens = 0
low.kv_tokens = 0
low.num_preemptions += 1
```

high 完成并释放 KV 后，low 重新进入并最终完成。

## 实验四：FCFS 与 Priority

FCFS 的 waiting 顺序按到达时间；KV 不足时牺牲 running 尾部。Priority 按
`(priority, arrival, request_id)`，数值越小越优先；KV 不足时选择排序键最差的 running
请求。

建议再加入持续到达的 priority=0 请求，观察低优先级请求可能长期等待。教学模型没有
aging 机制，这能直接展示 priority 的 starvation 风险。

## 与上游源码的对应关系

| 教学模型 | 上游源码 |
|---|---|
| `Request.deficit` | `num_tokens_with_spec + placeholders - num_computed_tokens` |
| `_schedule_running` | `Scheduler.schedule` 的 RUNNING 循环 |
| `_schedule_waiting` | `Scheduler.schedule` 的 WAITING 循环 |
| `_waiting_order` | `FCFSRequestQueue` / `PriorityRequestQueue` |
| `_fit_running_allocation` | `KVCacheManager.allocate_slots` + preemption retry |
| `_preempt` | `Scheduler._preempt_request` |
| `schedule` 中 computed 增量 | `Scheduler._update_after_schedule` |
| `update_from_output` | 同名上游方法的最小化版本 |

教学模型用 token 数近似 KV 占用，没有 block size、prefix cache、CoW、connector、encoder、
speculative token、async placeholder、deferred free 或 GPU 时间。因此实验验证的是状态机与
预算关系，不能替代上游单元测试或 NVIDIA 性能实验。

## 本机结果

7 个第 04 章标准库单元测试通过。真实 vLLM Scheduler 的 GPU 执行、PagedAttention block
布局、CUDA Graph batch shape、在线 TTFT/TPOT 和 preemption 性能成本尚未验证，因此章节
状态保持 `draft`。
