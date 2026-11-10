# 第 07 章实验：从 SchedulerOutput 到 ModelRunnerOutput

2026-10-08 复跑：本章 CPU 教学程序与测试结果见
[全项目文档复核记录](../meta/full-document-review-2026-10-08.md)。以下历史观察保留原日期；
待执行的 GPU 步骤仍是实验方案，不表示已经运行。

2026-09-08 全书复核补记：本地 CPU 重跑的版本、环境、命令与结果统一记录在
[全书检查报告](../meta/full-book-review-2026-09-08.md)。下文历史输出保留原日期；GPU
实验表格是待执行模板，空白不代表零值或测试通过。

本实验绑定 vLLM commit `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`，目标是分层验证执行栈，
而不是只记录一次端到端耗时。

## A. 无 GPU 契约实验

```bash
python3 examples/ch07_execution_pipeline.py
python3 -m unittest tests.test_ch07_execution_pipeline -v
```

检查所有 Worker 同 step、unique reply、ragged boundaries、execute/sample state 与 sampling 候选。

## B. 静态确认 Executor 与 Runner

记录 `distributed_executor_backend`、Executor/Worker class、环境变量是否显式设置、
实际 Runner class（现在只可能是 MRV2 的 `GPUModelRunner`）。同时记录 `VLLM_USE_V2_MODEL_RUNNER`
是否被设置：设为 `1` 只会警告，设为其他值会在 `Platform.validate_environ` 抛 `ValueError`。

## C. 启动与显存会计

记录硬件/软件、model revision、dtype/quantization、memory 参数、graph mode、TP/PP/DP，以及：

```text
device total memory
initial free memory
requested memory = total memory * gpu_memory_utilization
model weight memory
profile non-KV memory
CUDA Graph estimate and whether applied
available KV memory and block count
actual graph pool memory after capture
```

验证 `available KV = requested - non-KV - applied graph estimate`。若显式使用
`kv_cache_memory_bytes`，标记为手工容量，不与自动预算直接比较。

## D. MRV2 golden-output run

MRV1 已随 #63373 删除，不能再在同一份源码里做 MRV1/MRV2 对照（需要对照时必须分别检出 v0.33.0 与
`05caae03` 两份 vLLM，那是跨版本对照，不是同一 revision 内的实验）。本节只在 `05caae03` 上固定输出：

```bash
# MODEL 指向一个 MRV2 支持的模型；先完成预热，再保存结果。
MODEL=/path/to/model
vllm bench latency --model "$MODEL" --input-len 128 --output-len 32 --batch-size 1
```

以上是待在 GPU 上运行的延迟基线命令。`bench latency` 不会替你导出逐 token 的
正确性数据；下面的 token/logprobs 检查需要另行保存生成结果。

先用 greedy 保存 token/logprobs 作为 golden output，再增加 temperature/top-k/top-p（固定 seed）。分开记录
prepare、forward、sample、D2H、峰值显存与 block 数。若某个组合在配置阶段被 `_validate_model_runner` 拒绝，
应记录这条能力边界，不能删参数后宣称原组合已验证。

## E. Multiprocessing 与 PP

每 rank 记录 step sequence、rpc/global/local rank、device、execute/sample entry、是否 enqueue reply。
普通无输出聚合器路径预期所有参与 rank 执行、selected rank 提供回复；KV/encoder
connector 聚合器启用时会收集多 rank 回复再合并。

PP 额外记录 layer 范围、`IntermediateTensors` keys/shape/dtype、send/receive events、上一轮 send wait、
last-rank logits/sample 和 sampled-token broadcast。非末 rank 的空输出不是失败。

## F. 分离 step 时间

分别 trace RPC、request update、input preparation、block/slot preparation、attention metadata、forward、
logits、grammar、sampling、state postprocess、D2H、response serialization 和 scheduler update。CUDA events
测设备区间，monotonic clock 测 CPU 区间，并注明同步点。端到端 latency 不能命名为 kernel time。

## 结果模板

```text
source commit:
date:
hardware/software:
model/config:
executor / runner:
workload:

startup: init / load / profile / KV allocation / warmup-capture
steady step: update / prepare / forward / sample / D2H / scheduler
correctness: token match / logprob comparison / empty-output paths
limitations:
```

本机当前无可用 vLLM NVIDIA GPU runtime，因此这里只提供可复现实验协议；GPU 数字必须在真实环境填写。
