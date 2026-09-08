# 第 08 章实验：异步执行、编译与 CUDA Graph

2026-09-08 全书复核补记：本地 CPU 重跑的版本、环境、命令与结果统一记录在
[全书检查报告](../meta/full-book-review-2026-09-08.md)。下文历史输出保留原日期；GPU
实验表格是待执行模板，空白不代表零值或测试通过。

本实验绑定 vLLM commit `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`。目标不是只找一个最快
开关，而是分别测量调度重叠、编译、CUDA Graph、padding 和显存容量的影响。

## A. 无 GPU 契约实验

```bash
python3 examples/ch08_async_compile_graph.py
python3 -m unittest tests.test_ch08_async_compile_graph -v
```

确认 placeholder、stale output、配置模式、runtime mode、padding、fallback、稳定地址和 cache key。

## B. 环境记录

```text
source commit:
date:
GPU / driver / CUDA / PyTorch / vLLM:
model revision / dtype / quantization:
TP / PP / DP / EP / DBO:
runner: MRV1 or MRV2
optimization_level:
final CompilationConfig:
attention backend and AttentionCGSupport:
async_scheduling / max_concurrent_batches:
max_num_seqs / max_num_batched_tokens:
capture sizes / compile sizes / compile ranges:
```

必须记录最终解析后的配置及日志中的自动降级，不只记录命令行输入。

## C. 四组执行模式

至少比较：

```text
C0: compile NONE, cudagraph NONE
C1: VLLM_COMPILE, cudagraph NONE
C2: VLLM_COMPILE, cudagraph PIECEWISE
C3: VLLM_COMPILE, cudagraph FULL_AND_PIECEWISE
```

每组先冷缓存启动，再热缓存启动；完成固定 warmup 后再测稳定态。报告：

```text
startup total / compile / capture
graph memory / KV blocks / peak memory
request throughput / output token throughput
TTFT p50/p95/p99
TPOT p50/p95/p99
CPU schedule+prepare time
GPU forward and sample event time
FULL / PIECEWISE / NONE frequency
padding tokens and ratio
token/logprob correctness
```

## D. Shape 与 dispatch

构造小、中、大并发以及不同 prompt/decode 比例，收集 `CUDAGraphStat`。对每种 runtime batch 记录：

```text
actual num_tokens / num_reqs / query length
padded num_tokens / padded num_reqs
uniform decode
LoRA case / microbatch count
selected runtime mode
fallback reason
```

改变 capture sizes 后重复实验。命中率、padding、capture 时间和 graph memory 必须一起比较。

## E. Async scheduling

用完全相同的请求 arrival trace 分别启用和关闭 async scheduling。除吞吐和延迟外，记录：

```text
batch queue depth over time
in-flight batches and tokens
num_output_placeholders per request
preemptions with in-flight output
stale outputs
confirmed boundary
structured-output deferred samples
```

逐 token 比较输出。对于非 greedy sampling，固定 seed 并说明是否允许调度差异影响随机流。

## F. Nsight Systems / profiler 标注

时间线至少区分：

```text
EngineCore schedule
Worker request-state update
input and attention metadata preparation
H2D copy stream
model forward
sampler
D2H copy stream
copy event synchronize
scheduler update
```

检查 eager launch 序列与 graph replay、CPU 空洞、stream wait、意外 synchronize 和 graph fallback。

## G. MRV1 与 MRV2

只在两者都支持同一模型和功能组合时做性能对比。记录各自 descriptor、capture 数量和最终 runtime
mode。若一侧自动降级、禁用 async 或不支持某功能，将其列为能力差异，不把结果当同条件性能对比。

## H. 结果模板

```text
experiment id:
environment:
workload and arrival trace:
requested config:
resolved config:
cold/warm cache:

startup:
steady latency/throughput:
CPU/GPU timing:
graph mode and padding distribution:
memory/KV capacity:
correctness:
fallbacks/warnings:
interpretation:
limitations:
```

本机当前无可用 vLLM NVIDIA GPU runtime，因此只完成了 CPU 契约测试和可复现实验设计；任何真实
CUDA Graph 性能数字必须在目标 GPU 环境补录。
