# 第 02 章请求生命周期实验记录

2026-09-08 全书复核补记：本地 CPU 重跑的版本、环境、命令与结果统一记录在
[全书检查报告](../meta/full-book-review-2026-09-08.md)。下文历史输出保留原日期；GPU
实验表格是待执行模板，空白不代表零值或测试通过。

日期：2026-09-07

## 目标

在没有 PyTorch、CUDA 和可运行 vLLM 环境的机器上，用依赖为零的教学模拟器验证第
02 章最关键的对象边界和生命周期不变量。

本实验不验证 vLLM 性能，也不声称执行了真实 GPU 路径。

## 环境

```text
OS: macOS 26.6.2 (25G83)
Architecture: arm64
Python: 3.14.3
PyTorch: unavailable
CUDA: unavailable
vLLM source commit: 5893426b88f7b3cd21101d194eb1c6f0a6f0e27b
```

## 命令

```bash
python3 examples/ch02_request_lifecycle.py
python3 -m unittest tests/test_ch02_request_lifecycle.py
```

## 观察结果

```text
offline final outputs:
  req-A: 它让推理 (length)
  req-B: 它让推理更高效。 (length)
online stream:
  step 1: ['它']
  step 2: ['让']
  step 3: ['推理']
  abort: [('', 'abort')]
```

```text
.....
----------------------------------------------------------------------
Ran 5 tests in 0.000s

OK
```

## 验证的不变量

| 不变量 | 验证方式 |
|---|---|
| 离线路径循环 step 直到全部完成 | 两个不同输出长度请求均得到 final output |
| 在线路径交付增量 token | 连续三个 tick 分别得到“它”“让”“推理” |
| Core 使用内部 ID，用户看到外部 ID | abort 使用 internal ID，输出仍为 `chatcmpl-2` |
| 前端先注册本地状态 | submit 返回后同时检查 frontend/core 两侧字典 |
| prompt 与输出长度受统一上限约束 | 超过 `max_model_len` 时抛出 `ValueError` |

## 与真实源码的对应关系

| 教学模拟器 | 当前 vLLM 源码 |
|---|---|
| `Renderer.render` | `BaseRenderer.render_cmpl` / `render_chat` |
| `InputProcessor.process` | `InputProcessor.process_inputs` |
| `EngineCoreRequest` | `vllm/v1/engine/__init__.py::EngineCoreRequest` |
| `CoreRequestState` | `vllm/v1/request.py::Request` |
| `EngineCore.schedule` | `Scheduler.schedule` |
| `EngineCore.execute_model` | `Executor.execute_model` |
| `EngineCore.update_from_output` | `Scheduler.update_from_output` |
| `OutputProcessor.process` | `OutputProcessor.process_outputs` |

## 限制

- 没有真实 tokenizer、model forward、KV Cache、ZMQ 或 GPU。
- 调度器固定每请求每 step 一个 token，不模拟 token budget 和 preemption。
- abort 是确定性顺序，不模拟 GPU forward 与取消的真实竞争。
- 在线 collector 使用简化 deque，不模拟 vLLM 的 DELTA 合并实现。

真实动态验证需要 NVIDIA GPU 环境，至少覆盖 offline in-process、online
`AsyncMPClient`、单 rank `UniProcExecutor` 和多 rank `MultiprocExecutor` 四种组合。
