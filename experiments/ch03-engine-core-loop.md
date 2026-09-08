# 第 03 章实验记录：EngineCore 控制流模型

2026-09-08 全书复核补记：本地 CPU 重跑的版本、环境、命令与结果统一记录在
[全书检查报告](../meta/full-book-review-2026-09-08.md)。下文历史输出保留原日期；GPU
实验表格是待执行模板，空白不代表零值或测试通过。

## 目的

在没有 CUDA、PyTorch 和真实模型权重的环境中，验证第 03 章依赖的控制流不变量：

1. client 选择由 multiprocessing 与 asyncio 两个维度共同决定；
2. 普通 `step()` 虽以 `non_block=True` 发起执行，仍在同一次调用中等待 Future；
3. batch queue 先填充、后按 FIFO 结算，使多个 batch 同时在途；
4. 模型执行期间到达的 abort 必须在 `update_from_output` 之前生效；
5. `keep` pause 冻结调度，resume 后请求继续；
6. executor failure 是进程级致命事件，零秒 shutdown 则立即 abort。

## 运行命令

```bash
python3 examples/ch03_engine_core_loop.py
python3 -m unittest tests/test_ch03_engine_core_loop.py
```

## 预期演示输出

```text
tick=0 launched=True queue=1 outputs=None
tick=1 launched=True queue=1 outputs=(EngineCoreOutput(... token_ids=(100,) ...),)
tick=2 launched=True queue=1 outputs=(EngineCoreOutput(... token_ids=(101,) ...),)
tick=3 launched=False queue=0 outputs=(EngineCoreOutput(... token_ids=(102,), finish_reason='length'),)
```

第一轮只提交 batch 而没有请求输出；第二、三轮在提交新 batch 的同时结算更老的
batch；最后一轮不再提交工作，只排空队列。这正是“异步形状”与“真正保持多个在途
batch”的区别。

## 与上游源码的对应关系

| 教学模型 | 上游源码 |
|---|---|
| `select_client` | `EngineCoreClient.make_client` |
| `TeachingEngineCore.step` | `EngineCore.step` |
| `step_with_batch_queue` | `EngineCore.step_with_batch_queue` |
| `aborts_queue` | `EngineCore.aborts_queue` / `_process_aborts_queue` |
| `TeachingEngineCoreProc.run_once` | `EngineCoreProc.run_busy_loop` 的一次迭代 |
| `MessageType` | `EngineCoreRequestType` |
| `ShutdownState` | `EngineShutdownState` |

教学模型没有模拟真实 Scheduler 的 token budget、KV block、structured output、推测
解码、DP wave、ZMQ 序列化与 GPU 并发。因此它只能验证控制流解释，不能替代上游测试
或真实 GPU 性能实验。

## 结果

本机 Python 标准库测试通过。真实 vLLM 初始化、GPU Future 完成时序、进程信号和
NVIDIA trace 尚未验证，因此章节状态保持 `draft`。
