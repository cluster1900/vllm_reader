---
title: "第02章 vLLM 全景与一次请求的生命周期"
status: draft
source_repository: vllm-project/vllm
source_path: ../vllm
source_commit: 05caae03c20f69436ff7ec23ded339ea104f1a01
source_branch: main
source_dirty: false
verified_at: 2026-11-10
content_complete: true
runtime_verified: false
audience: "初中级程序员和软件工程类学生；具备 Python 基础，不要求推理系统背景"
pedagogy_reviewed_at: 2026-10-08
scope: "V1 文本生成主链；覆盖离线 LLM.generate 和 OpenAI-compatible 在线 chat 入口"
prerequisites:
  - 第01章
---

# 第02章 vLLM 全景与一次请求的生命周期

> 调用 `LLM.generate()` 或发送一条 HTTP 请求后，数据究竟经过哪些对象、线程、
> 进程和设备？

## 本章定位

第 01 章解释了一个 decoder-only Transformer 怎样预测下一个 token。本章把镜头拉远：
模型只是 vLLM 的一个执行部件。一个可用的推理系统还要接收请求、渲染 chat template、
做 tokenization、检查参数、管理并发、选择当前要算的 token、组织 GPU 输入、把 token
重新解码成文本，并在客户端断开时及时释放状态。

本章先建立全书最重要的一张地图。后续章节会分别放大其中的局部：

- 第 03 章：Engine、EngineCore、IPC 和主循环。
- 第 04 章：Scheduler 与 Continuous Batching。
- 第 05 章：KV Cache block 的生命周期。
- 第 06 章：block table 怎样变成 attention 的真实寻址。
- 第 07 章：Executor、Worker、Model Runner 和 Sampler。

本章以当前固定 revision 的 **V1 文本生成路径**为准。多模态、pooling、beam search、
structured output、speculative decoding 和分离式 prefill/decode 只在它们改变主链边界时
指出，不展开内部算法。

本章绑定源码 revision `05caae03c20f69436ff7ec23ded339ea104f1a01`（vLLM `main`，提交日期 2026-11-09，美国太平洋时间；2026-11-10 核对源码引用与行号）。

## 阅读目标

读完本章后，你应该能够：

1. 分清 offline `LLM.generate()` 和 online OpenAI-compatible API 的前端差异。
2. 解释 Renderer、InputProcessor、Engine client、EngineCore、Scheduler、Executor、
   OutputProcessor 各自负责什么。
3. 画出一个请求从 JSON 或 Python 对象到 GPU，再回到文本的完整路径。
4. 区分 `EngineInput`、`EngineCoreRequest`、`Request`、`SchedulerOutput`、
   `ModelRunnerOutput`、`EngineCoreOutput` 和 `RequestOutput`。
5. 指出哪些边界是普通函数调用，哪些是 asyncio queue、线程队列、ZMQ 或 GPU 执行。
6. 解释客户端取消后 abort 为什么必须同时清理前端状态和 EngineCore 状态。
7. 不再把 `LLMEngine`、`AsyncLLM`、`EngineCore`、Worker 和模型混成一个“引擎”。

## 如何阅读本章

这一章最容易出现的问题不是数学太难，而是类名太多。不要尝试一次记住所有对象；先把
一次请求当成一张工单，观察它如何经过“接单、标准化、排队、执行、交付”五个阶段。
之后再把 Renderer、EngineCore、Scheduler、Worker 和 OutputProcessor 放回对应阶段。

第一次阅读只追一条普通文本生成的 happy path，看到多模态、structured output、DP
或异常分支时先知道它挂在哪里即可。第二次阅读再区分函数调用、线程队列、ZMQ 和 GPU
边界，因为同一个 Python 方法名跨过这些边界后，性能和失败传播方式会完全不同。

本章不要求你理解 attention kernel。你需要建立的是系统地图：谁拥有请求状态，谁拥有
调度权，谁拥有设备内存，谁把 token 还原成用户看到的文本。后续章节都是对这张地图的
局部放大。

**先分清三种并发对象。** 进程拥有自己的内存空间；线程共享所在进程的内存；
asyncio task 是事件循环调度的一段协程，在 `await` 等待时允许其他 task 继续。
ZMQ 负责进程间 IPC 消息传递，SSE 负责向前端 HTTP 客户端持续推送事件；它们解决的是系统层通信与异步并发，不参与模型内部张量的数值并行计算。

**第一遍走法：** 读 2.1–2.4、2.6–2.12，再运行 2.17 的模拟器；用 2.14 检查取消路径。
**停下来追：** token ID 已从 GPU 返回，为什么用户还没看到文字？先查 OutputProcessor
的增量解码，再查 collector 与 SSE；模型输出不直接等于 HTTP 输出。

## 2.1 为什么模型外面还需要一套系统

只看单个请求，推理循环似乎很简单：

```text
prompt -> tokenize -> model.forward -> sample -> decode -> text
```

但服务面对的是持续到达、长度不同、参数不同的许多请求。一个请求可能有 2,000 个
prompt token，只生成 8 个 token；另一个 prompt 很短，却要生成 1,000 个 token。它们
到达时间不同，也可能中途取消。如果每个请求独占 GPU 直到结束，就会出现三个问题：

1. **GPU 利用率低。** Decode 每步通常只有少量新 token，单请求难以填满 GPU。
2. **KV Cache 浪费。** 按最大长度为每个请求连续预留显存，会产生大量空洞。
3. **延迟失控。** 一个长请求会阻塞后到达的短请求，流式响应和取消也很难处理。

vLLM 的系统价值可以先压缩为一句话：

> 前端把不同形式的用户请求统一成内部请求；EngineCore 在每个 step 重新决定算谁；
> 执行层把这一轮 token 工作变成 GPU 计算；输出层再把增量 token 还原成用户协议。

```mermaid
flowchart LR
    U[用户输入] --> F[Frontend\n协议与输入处理]
    F --> C[EngineCore\n调度与资源协调]
    C --> E[Executor / Worker\n模型执行]
    E --> G[(GPU)]
    G --> E
    E --> C
    C --> O[OutputProcessor\n增量解码与完成判断]
    O --> U2[Python 对象或 SSE]
```

> **读图方法：** 阅读“为什么模型外面还需要一套系统”这张流程图时，先把方框看成对象或状态，把箭头看成数据或控制的移动。第一遍从左向右建立顺序，第二遍再核对分支发生的条件。

这张图有两个重要含义：

- 模型并不理解 HTTP、chat message、SSE 或用户提供的 request ID。
- API Server 不直接决定 GPU 当前计算哪些 token；这个决定属于 Scheduler。

## 2.2 三层心智模型：协议面、控制面、数据面

为了避免被类名淹没，可以先把整个系统分成三层。

### 协议面

协议面面对用户，处理 Python API、HTTP、OpenAI-compatible schema、chat template、
tokenization、流式 JSON、tool/reasoning parser 等问题。

典型对象：

- `LLM`
- `OpenAIServingChat`
- `OnlineRenderer` / `BaseRenderer`
- `InputProcessor`
- `OutputProcessor`

### 控制面

控制面维护请求状态和资源决策。它传递的主要是 request metadata、token 预算、block
ID、完成状态，而不是模型的大型 activation tensor。

典型对象：

- `EngineCoreClient`
- `EngineCore`
- `Scheduler`
- `KVCacheManager`
- `SchedulerOutput`

### 数据面

数据面把调度结果变成 device tensor 和 kernel 执行，包括模型权重、hidden states、
KV Cache、attention、sampling 以及必要的 collective。

典型对象：

- `Executor`
- `Worker`
- `GPUModelRunner`
- 模型 `torch.nn.Module`
- attention backend 和 CUDA/Triton kernel

### 生活化类比：现代餐厅运作系统

初学者在第一次面对 `Renderer`、`InputProcessor`、`EngineCoreClient`、`EngineCore`、`Scheduler`、`Worker`、`ModelRunner`、`OutputProcessor` 等十几个类时，极易陷入“类名迷宫”。为了建立稳固的第一直觉，可以把 vLLM 想象成一家高度工业化的现代中餐厅：

| 餐厅角色与运作流程 | vLLM 对应组件 | 在系统中所处的责任面 |
|---|---|---|
| **顾客点单与菜品清单** | 用户 HTTP API / OpenAI 格式输入 | 外部输入 |
| **前厅接待与点菜员** | `Renderer` & `InputProcessor` | **协议面**：把顾客口语化的点单转换成后厨能认的标准工单（`EngineCoreRequest`），并校验规格与忌口 |
| **前厅对讲机 / 传菜铃** | `EngineCoreClient`（IPC 通信） | **协议与控制面边界**：在多进程路径传递工单；同步/异步等待方式取决于 client 类型 |
| **后厨调度总管 / 领班** | `EngineCore` & `Scheduler` | **控制面**：根据请求进度、token 预算与 KV 容量，在每一个出菜节拍决定本轮计算哪些请求（`SchedulerOutput`） |
| **调料库管员** | `KVCacheManager` | **控制面**：管理公用酱料与已腌制好的半成品（KV Block），告诉总管还能不能接新菜、哪些桌可以复用老汤（Prefix Cache） |
| **各灶台大厨工位** | `Executor` & `Worker` | **数据面**：管理特定的一台或多台燃气灶（GPU），负责提前把锅烧热、准备设备执行环境 |
| **炒锅与装盘托盘** | `GPUModelRunner` | **数据面**：把总管派来的单子倒进锅里，装配成火候所需的数据结构（Tensor），驱动锅内翻炒（Model Forward） |
| **出菜检验打包装盒员** | `OutputProcessor` | **协议面交付层**：从后厨接过刚出锅的菜，核对是否上齐，把 token 逐个转换回人类可读的文字，流式交付给顾客（SSE 响应） |

类比说明职责，不代表真实对象一一对应进程：Model Runner 负责组织计算，模型与 GPU
算子负责数值运算；OutputProcessor 生成用户层输出，SSE 格式由 API 层包装。

### 软件工程基础：进程、线程与协程边界

在深入具体类之前，软件工程类学生必须牢固区分三类并发抽象，因为它们在 vLLM 中承担着完全不同的职责：

- **进程**：拥有独立的虚拟地址空间；多个进程仍可映射同一共享内存。常见启用 GIL
  的 CPython 部署中，各进程的解释器锁独立，能减少前后端争用；资源竞争和故障传播
  仍可能跨进程发生，不能保证后端崩溃对服务没有影响。
- **线程**：共享所在进程的地址空间，但有自己的执行栈。启用 GIL 的 CPython 中，
  同一解释器通常不能让多个线程同时执行 Python 字节码；释放 GIL 的扩展与阻塞 IO
  可以重叠。不能把这个限制推广到所有 Python 实现或 free-threaded 构建。
- **异步协程/task**：在事件循环内协作推进，等待尚未就绪的操作时可让出执行机会。
  切换 task 不需要每次都切换 OS 线程，但事件循环仍有调度成本；并非遇到任意
  `await` 都一定让出执行。直接在协程中执行长时间同步计算仍会阻塞该事件循环。

[Python 官方说明](https://docs.python.org/3/howto/free-threading-python.html)区分了启用
GIL 与 free-threaded 构建；本书常见部署的 GIL 解释需要带上这个前提。

```mermaid
flowchart TB
    subgraph P[协议面 CPU]
        API[Python / HTTP API]
        R[Renderer]
        IP[InputProcessor]
        OP[OutputProcessor]
    end
    subgraph C[控制面 CPU]
        CL[EngineCoreClient]
        EC[EngineCore]
        S[Scheduler]
        KVM[KV Cache metadata]
    end
    subgraph D[执行侧: CPU 派发与 GPU 计算]
        EX[Executor]
        W[Worker / Model Runner]
        M[Model + Sampler]
        KV[(GPU KV tensors)]
    end
    API --> R --> IP --> CL --> EC
    EC --> S
    S <--> KVM
    S -->|调度结果交回 EngineCore| EC
    EC --> EX --> W --> M
    M <--> KV
    M --> EC --> CL --> OP --> API
```

> **读图方法：** 这张图用于压缩“数据面”的整体关系。先从上向下找到起点、关键转换和终点，再问每条跨层箭头是否意味着函数调用、消息传递、内存访问或状态更新。

注意：这是**职责分层**，不等于固定的进程分层。`UniProcExecutor` 会让逻辑 Worker
对象与 EngineCore 位于同一个操作系统进程；`MultiprocExecutor` 或 Ray 才会把 Worker
放到其他进程。后文会专门修正这一点。

## 2.3 两个主要入口

> **本节先看：** 本节要回答：**两个主要入口**。下面的小节会逐层拆开概念、运行过程和源码落点；第一次阅读先抓对象之间的关系，第二次再记类名、字段和分支。

### 离线入口：`LLM.generate()`

离线推理适合脚本、评测、批处理和数据生成。调用者提交一组 prompt，函数通常等到
全部请求结束后返回 `list[RequestOutput]`。

```python
from vllm import LLM, SamplingParams

llm = LLM(model="your-model")
outputs = llm.generate(
    ["Explain KV cache", "Explain continuous batching"],
    SamplingParams(temperature=0, max_tokens=64),
)
```

当前源码中的主干是：

```text
LLM.generate
  -> OfflineInferenceMixin._run_completion
  -> OfflineInferenceMixin._add_completion_requests
  -> Renderer.render_cmpl
  -> LLMEngine.add_request
  -> OfflineInferenceMixin._run_engine
       while has_unfinished_requests():
           LLMEngine.step()
```

`LLM.generate()` 自己没有写模型循环。它先检查 runner type、补默认
`SamplingParams`，然后交给 `_run_completion()`。`_run_engine()` 才持续调用
`LLMEngine.step()`，收集完成项，最后按数值 request ID 排序，保证返回顺序与输入顺序
一致，即使短请求先完成。

[源码] `vllm/entrypoints/llm.py` - `LLM.generate`

v0.32.0 将旧的 offline_utils 模块移到 `vllm/entrypoints/common/offline.py`，
不是删除离线调用链；`LLM` 仍继承 `OfflineInferenceMixin`，上述方法仍在新模块。

[源码] `vllm/entrypoints/common/offline.py` -
`OfflineInferenceMixin._run_completion`、`_run_engine`

### 在线入口：OpenAI-compatible chat API

在线服务面对的是 HTTP 请求和异步并发。以 chat completion 为例，入口接收到
`ChatCompletionRequest` 后，要先校验模型、渲染消息、计算 `max_tokens`、构造
`SamplingParams`，再从 `AsyncLLM.generate()` 得到一个异步生成器。

```text
POST /v1/chat/completions
  -> OpenAIServingChat.create_chat_completion
  -> OpenAIServingChat._create_chat_completion
  -> OnlineRenderer.render_chat
  -> AsyncLLM.generate
  -> per-request RequestOutputCollector
  -> stream generator / full generator
  -> JSON or data: ...\n\n
```

当 `stream=true` 时，API 层遍历 `RequestOutput`，组装
`ChatCompletionStreamResponse`，最终写成 SSE：

```text
data: {"id":"chatcmpl-...","choices":[...]}

data: [DONE]
```

这里有两层“流”：

1. `AsyncLLM.generate()` 产生 vLLM 内部的 `RequestOutput`。
2. `chat_completion_stream_generator()` 把它变成 OpenAI-compatible SSE chunk。

模型和 EngineCore 都不负责 JSON 序列化。

v0.33.0 把第 2 层再拆成两段：`chat_completion_stream_generator_raw()`
（`vllm/entrypoints/openai/chat_completion/serving.py:510`）只负责读 `RequestOutput`、
交给 reasoning/tool 解析（每个 choice 一个 `ChatStreamProcessor`），产出 `ChatCompletionStreamResponse`
或 `ErrorResponse` **对象**，不碰 SSE 文本；外层的 `chat_completion_stream_generator()`（`:473`）
再把每个对象 `model_dump_json()` 成 `data: ...\n\n`，序列化出错时输出一条错误 chunk，
最后无论成败都补 `data: [DONE]`。Cohere 与 Anthropic 端点复用同一个 raw 生成器、自行序列化，
所以它们的流式错误现在能保留各自的格式（[5be4266f5](https://github.com/vllm-project/vllm/commit/5be4266f5)，
[#61947](https://github.com/vllm-project/vllm/pull/61947)）。非流式的
`chat_completion_full_generator()`（`:966`）也不再自己写 tool-choice 分支，而是调用
`assemble_chat_message_fields()`（`vllm/entrypoints/openai/chat_completion/message_assembly.py:26`），
与流式、批量 derender 共用同一套组装规则（[e879223b5](https://github.com/vllm-project/vllm/commit/e879223b5)，
[#63148](https://github.com/vllm-project/vllm/pull/63148)；
[5ea2c7ff1](https://github.com/vllm-project/vllm/commit/5ea2c7ff1)，
[#62706](https://github.com/vllm-project/vllm/pull/62706)）。

```mermaid
flowchart LR
    GEN["AsyncLLM.generate: RequestOutput 流"] --> RAW["chat_completion_stream_generator_raw: reasoning/tool 解析, 产出 ChatCompletionStreamResponse 或 ErrorResponse"]
    RAW --> SER["chat_completion_stream_generator: model_dump_json 并加 data: 前缀"]
    SER --> SSE["SSE: data: JSON 若干条, 最后 data: [DONE]"]
    RAW -.-> OTH["Cohere / Anthropic 端点: 自行序列化同一个对象流"]
    FULL["chat_completion_full_generator"] --> ASM["assemble_chat_message_fields: 非流式消息字段"]
```

> **读图方法：** 这是 chat 流式输出两段式结构的流程图。先沿实线从左到右看一条 SSE 如何由 `RequestOutput` 变成 `data:` 帧；虚线表示其他协议端点复用 raw 生成器，下方一条独立链路是非流式路径使用的消息组装函数。

[源码] `vllm/entrypoints/openai/chat_completion/serving.py` -
`OpenAIServingChat.chat_completion_stream_generator`、`chat_completion_stream_generator_raw`、
`chat_completion_full_generator`

[源码] `vllm/entrypoints/openai/chat_completion/serving.py` -
`OpenAIServingChat._create_chat_completion`、`chat_completion_stream_generator`

[源码] `vllm/v1/engine/async_llm.py` - `AsyncLLM.generate`

> **已在本书基线中、未在正文展开的变化：** 以下提交都是本书基线 `05caae03` 的祖先（且都不在 v0.33.0 `4ba787ec` 中）；本章正文对它们尚未展开，仍按 v0.33.0 时代的写法叙述（Model Runner V1 删除除外）。条目里 `@` 后的短 SHA 是各提交当时的版本，行号不是 `05caae03` 上的行号。这些条目来自 vLLM main 2026-10-28～10-29（UTC；行号取自 10-30 的 `2eae56c4b`）。
> - [9fc019a749](https://github.com/vllm-project/vllm/commit/9fc019a749cbf3235cc06aa5357b17fd37cb648a)（[#64097](https://github.com/vllm-project/vllm/pull/64097)）：`run_batch` 支持 `/v1/completions`（`entrypoints/launchers/run_batch.py` L182–L197、L828–L829 @ `36d4ef654`），不再仅限 chat completions。
> - [f60edccd00](https://github.com/vllm-project/vllm/commit/f60edccd0079428503f9e9f66e3b651aa4da73db)（[#64177](https://github.com/vllm-project/vllm/pull/64177)）：chat completions 序列化省略 null 的 `annotations` 与 `usage.prompt_tokens_details`（这两个字段各自用 `Field(default=None, exclude_if=lambda v: v is None)`，不是 `exclude_none`；`vllm/entrypoints/openai/chat_completion/protocol.py`、`vllm/entrypoints/serve/engine/protocol.py`）。
> - main 2026-10-29（UTC）：[5c1939140](https://github.com/vllm-project/vllm/commit/5c19391400c58ae10feca2483f55c547df211715)（[#63896](https://github.com/vllm-project/vllm/pull/63896)）：`ResponsesRequest.thinking_token_budget` 写入 `SamplingParams`（`vllm/entrypoints/openai/responses/protocol.py` L452、L650 @ `2eae56c4b`），与 chat completions 对齐。Python 前端的 `/v1/responses` 会把该字段写入 `SamplingParams`；Rust 前端在本书基线里已经提供 `/v1/responses`（#63902，`eb409c9aa`），但转换时把 `thinking_token_budget` 固定为 `None`（`rust/src/server/src/routes/openai/responses/convert.rs:232`）。

### 两条路径的共享与分叉

> **本节先看：** 下面先用表格整理“两条路径的共享与分叉”。先横向比较每一列解决的问题和适用边界，再把具体名称映射到源码。

| 问题 | 离线 `LLM.generate` | 在线 chat serving |
|---|---|---|
| 用户输入 | Python prompt/list | HTTP JSON messages |
| chat template | `LLM.chat` 时使用 | `OnlineRenderer.render_chat` |
| 请求推进 | 前端循环 `LLMEngine.step()` 取结果；默认后端有独立 busy loop | EngineCore 后台 busy loop |
| 常见 client | 默认 `SyncMPClient`；显式关闭 MP 时为 `InprocClient` | `AsyncMPClient` 或 DP 变体 |
| 输出交付 | 返回最终 `list[RequestOutput]` | 每请求异步 collector，再转 SSE/JSON |
| 取消来源 | 调用方显式 abort | HTTP 断开、task cancel、显式 abort |
| 共同后端 | EngineCore、Scheduler、Executor、Worker、Model Runner | 同左 |

```mermaid
flowchart LR
    subgraph OFF[Offline]
        PY[LLM.generate] --> OR[Renderer]
        OR --> LE[LLMEngine]
    end
    subgraph ON[Online]
        HTTP[OpenAI HTTP] --> OAI[OpenAIServingChat]
        OAI --> ORE[OnlineRenderer]
        ORE --> AL[AsyncLLM]
    end
    LE --> CC[EngineCoreClient]
    AL --> CC
    CC --> CORE[EngineCore]
    CORE --> EXEC[Executor / Worker / GPU]
    EXEC --> CORE
    CORE --> CC
    CC --> OP[OutputProcessor]
    OP --> RET[最终列表或流式输出]
```

> **读图方法：** 这是“两条路径的共享与分叉”的流程图。先从左向右只追一条主路径，确认输入经过哪些关键阶段到达输出；第二遍再看虚线、回边和旁路，它们通常表示反馈、复用或可选分支。

### Decisions API：把生成换成结构化判别

> **本节先看：** 聊天接口要模型「写出」答案；Decisions / System One 要模型对预先给定的选项打分。实现上仍可能走到引擎的 generate 或分类路径，但 HTTP schema、handler 与「一次请求含多道题」的编排都在独立模块里。这不是与离线 `LLM.generate`、在线 chat 并列的第三条主链，而是协议面上另外两条入口。

**两条入口。** 都在 `register_api_routers` 里注册（`vllm/entrypoints/launchers/api_server/routers.py:28-38` @ `05caae03`）。这组接口随 #60465（`/v1/decisions`，`c923e7769`）、#59299（`/v1/systemone`，`d33872dfe`）、#62437（joint read，`7e3d55956`）进入，均已包含在本书基线中。

1. `POST /v1/decisions`（`openai/decisions/api_router.py:24-40` @ `05caae03`）——OpenAI Decisions 形状，只接受文本输入（协议层会拒绝 `input_image`，`protocol.py:31-45`）。请求里是 `input` 和 `questions[]`；每题 `type` 为 `predicate`、`choice` 或 `score`（`protocol.py:59-91`；`choice` 至少两个互异选项，`score` 至少两个等级）。`OpenAIServingDecisions.create_decisions`（`serving.py:55-83` @ `05caae03`）做模型检查、题目数上限和 admission，再调用策略的 `read` 或 `read_joint`。没有挂上 handler 时（没有对应 `ReadStrategy`，或策略任务不是 `generate`），入口返回 501（`api_router.py:40-48`；`create_error_response` 把 `NotImplementedError` 映射为 501）。
2. `POST /v1/systemone`（`generate/structured_decisions/api_router.py:33-45` @ `05caae03`）——TypeSafe System One，由 `ServingStructuredDecisions` 处理。它把「状态」和多道 typed 问题编成一次或多次读取。对 `generate` 任务的策略（如 `NextTokenStrategy`），每次读取取回复首 token 在标签集合上的 logprobs，再对标签做 softmax 得到概率；这类策略需要 `--logprobs-mode raw_logprobs` 或 `processed_logprobs`（`strategies.py:1483-1486` @ `05caae03`）。模式不对并不会让服务启动失败：`init_decisions_state` 会吞掉 `select_read_strategy` 抛出的 `ValueError`（`state.py:26-38` @ `05caae03`），两个入口随后都返回 501。有的模型走 `classify` 或 `token_classify` 专用头，不经过「首 token logprobs」这条路径，细节见官方文档。官方文档也写明：有对应策略的生成模型默认会挂上该端点；没有策略时同样返回 501。

白话解释：这里的 softmax 是把各标签的 log-probability 先变成相对大小，再归一化成加起来为 1 的一组概率，而不是再做一次自由文本生成。

```mermaid
flowchart LR
    C[客户端] --> D["POST /v1/decisions"]
    C --> S["POST /v1/systemone"]
    D --> H1[OpenAIServingDecisions]
    S --> H2[ServingStructuredDecisions]
    H1 --> STR[ReadStrategy.read / read_joint]
    H2 --> STR
    STR --> ENG[EngineClient / AsyncLLM]
    ENG --> OUT[答案 + probabilities + usage]
```

> **读图方法：** 左右两个 HTTP 入口在策略层汇合，都依赖 EngineClient；差别在 schema，以及如何把题目编成模型输入。模型没有对应 ReadStrategy 时，这两个入口都是 501，不要误当成聊天接口的 404。`joint=true` 只有声明了 `joint_reads` 的策略（如 DiffusionGemma）才会走 `read_joint`，否则是 400。

**和 chat 主链的差别（初读只需记住这三句）：**

- 输出是结构化字段（选项 / 分数 / 置信度），不是自由文本 token 流。
- 一道请求可含多题；是否 joint read 取决于策略与模型。
- 概率会受同一 batch 里的其他请求以及 prefix cache 影响；要可复现时，官方文档提示设置 `VLLM_BATCH_INVARIANT=1`。

[源码] `vllm/entrypoints/openai/decisions/api_router.py` - `create_decision`（`:40` @ `05caae03`）

[源码] `vllm/entrypoints/openai/decisions/serving.py` - `OpenAIServingDecisions.create_decisions`（`:55` @ `05caae03`）

[源码] `vllm/entrypoints/openai/decisions/protocol.py` - `DecisionRequest`、`PredicateQuestion`、`ChoiceQuestion`、`ScoreQuestion`

[源码] `vllm/entrypoints/generate/structured_decisions/api_router.py` - `/v1/systemone`（`:33` @ `05caae03`）

[源码] `vllm/entrypoints/generate/structured_decisions/serving.py` - `ServingStructuredDecisions`（`:56` @ `05caae03`）

[文档] `docs/models/structured_decisions_models.md`


## 2.4 Renderer 与 InputProcessor 不是同一个阶段

初学者常把“输入处理”理解成一个 tokenize 函数。当前 V1 实际上有两层。

### Renderer：从用户语义到 `EngineInput`

Renderer 处理更靠近用户协议的工作：

- 解析字符串、token IDs、chat messages 或多模态输入。
- 应用 chat template。
- 调用 tokenizer。
- 执行多模态 processor，生成 placeholder 和 feature metadata。
- 给处理后的输入附加 arrival time。

`BaseRenderer.render_cmpl()` 的骨架是：

```text
render_prompts
  -> tokenize_prompts
  -> apply prompt extras
  -> process_for_engine
  -> EngineInput
```

`render_chat()` 多一步 `render_messages()`，先把 role、content、tools 等消息结构变成
模型可读的 prompt。

v0.33.0 在在线 Renderer 里有三处会改变 chat prompt 的行为：

- `--reasoning-effort-map`（JSON 对象）会被传入 `OnlineRenderer(reasoning_effort_map=…)`
  （`vllm/renderers/online_renderer.py:185`）。`render_chat` 构造 chat params 时先把请求里的
  `reasoning_effort` 按映射改写再交给 chat template（`build_chat_params(..., reasoning_effort_map)`，
  `:236`），Harmony 路径用同一个 `map_reasoning_effort()`（`vllm/renderers/params.py:53`）。
  它只改写 effort 字符串，不增加新的 effort 取值
  （[41d9d1e14](https://github.com/vllm-project/vllm/commit/41d9d1e14)，
  [#61094](https://github.com/vllm-project/vllm/pull/61094)）。
- `tool_strict_level` 的构造默认值从 `"auto"` 改为 `None`，未设置时回落到 tool parser 自己的默认值
  （[8fa5ae844](https://github.com/vllm-project/vllm/commit/8fa5ae844)，
  [#61328](https://github.com/vllm-project/vllm/pull/61328)）。
- 需要把 prompt 文本带回响应（例如 echo，`params.needs_detokenization`）时，
  `BaseRenderer._tokenize_singleton_prompt` 先应用截断（`apply_post_tokenization`），再把截断后的 token
  反解为文本，保证返回文本与真正送进引擎的 token 一致
  （[d74f4180c](https://github.com/vllm-project/vllm/commit/d74f4180c)，
  [#61876](https://github.com/vllm-project/vllm/pull/61876)）。

下图把上面的骨架展开成一次渲染里的真实分支，回答一个问题：同一批 prompt 在 Renderer 里依次经过哪些关口，哪些只在特定条件下才发生。

```mermaid
flowchart TB
    IN["render_cmpl: prompts<br/>render_chat: conversations"] --> AT["记 arrival_time<br/>函数入口取一次，整批共用"]
    AT --> KIND{"入口"}
    KIND -- "render_cmpl" --> RP["render_prompts<br/>空列表抛 VLLMValidationError"]
    KIND -- "render_chat" --> RM["render_messages<br/>chat template 渲染 role、content、tools"]
    RP --> TK
    RM --> TK
    subgraph TK["tokenize_prompts: 逐个 prompt"]
        T0{"已有 token ids<br/>或 embeds?"}
        T0 -- "否，是文本" --> T1["pre-tokenization 截断<br/>tokenizer 编码"]
        T1 --> T2["按字符偏移修正"]
        T0 -- "是" --> T3
        T2 --> T3["post-tokenization 截断"]
        T3 --> T4{"needs_detokenization<br/>且还没有文本?"}
        T4 -- "是" --> T5["截断后的 token 反解为文本"]
        T4 -- "否" --> T6["保持原样"]
    end
    TK --> EX["_apply_prompt_extras<br/>render_chat 额外并入 media_io_kwargs"]
    EX --> PE{"process_for_engine<br/>有 encoder_prompt?"}
    PE -- "是" --> ED["_process_enc_dec<br/>分别处理 encoder 与 decoder 输入"]
    PE -- "否" --> SG{"带 embeds?"}
    SG -- "是" --> EM["_process_embeds"]
    SG -- "否" --> TM["_process_tokens<br/>有多模态数据则走 multimodal processor"]
    TM --> TR["_truncate_expanded_prompt<br/>仅多模态且设了 truncate_prompt_tokens"]
    ED --> OUT["EngineInput<br/>附 arrival_time"]
    EM --> OUT
    TR --> OUT
```

> **读图方法：** 第一遍只看最左的主干：入口取时间、进入 `render_prompts` 或 `render_messages`、逐个 tokenize、并入额外字段、`process_for_engine`、得到 `EngineInput`。
> 第二遍再看分支：`render_chat` 比 `render_cmpl` 多一步 `render_messages`，并多并入一个 `media_io_kwargs`；tokenize 框里，已经是 token ids 的输入跳过编码，
> 只有 echo 之类需要文本回显时才会在截断之后反解；最后一段按输入类型分三路，`_truncate_expanded_prompt` 只在输入是多模态、并且设了 `truncate_prompt_tokens` 时才会真正再截一次，因为一个 placeholder 可能展开成很多 token，把长度重新顶过上限（其它情况函数直接返回，`base.py:1108-1112`）。

图中每个框都对应固定 commit 里的真实函数（`vllm/renderers/base.py` @ `05caae03`）：`render_cmpl` `:1290`（入口取 `arrival_time` `:1298`）、`render_chat` `:1342`、
`render_prompts` `:553`（空列表检查 `:556-559`）、`render_messages` `:590`、`tokenize_prompts` `:835`、
`_tokenize_singleton_prompt` `:711-738`（编码前截断与偏移修正 `:722-727`，编码后截断 `:729`，截断后反解 `:733-738`）、`_apply_prompt_extras` `:862`
（`render_chat` 并入 `media_io_kwargs` 在 `:1369-1371`）、`process_for_engine` `:1246`（`encoder_prompt` 分支 `:1255`，`arrival_time` 写入 `:1262`）、
`_process_singleton` `:1154`（embeds 分支、调用 `_truncate_expanded_prompt` `:1168`；函数本身 `:1090`，仅多模态且设了 `truncate_prompt_tokens` 时才截，`:1108-1112`）、`_process_enc_dec` `:1197`。

渲染阶段的客户端错误也统一了类型：空 prompt、`multi_modal_uuids` 缺项或长度不符、
`prompt_embeds` 开关/形状错误，原先抛 `ValueError`，现在抛带 `parameter` 字段的
`VLLMValidationError`（[1cc332ee8](https://github.com/vllm-project/vllm/commit/1cc332ee8)，
[#62738](https://github.com/vllm-project/vllm/pull/62738)），因此会和 `InputProcessor` 的校验错误一样
被 `AsyncLLM` 当作客户端错误（见 2.14）。

[源码] `vllm/renderers/base.py` - `BaseRenderer.render_cmpl`、`render_chat`

[源码] `vllm/renderers/online_renderer.py` - `OnlineRenderer.render_chat`

[源码] `vllm/renderers/base.py` - `BaseRenderer.validate_token_ids`

### InputProcessor：从 `EngineInput` 到 `EngineCoreRequest`

`InputProcessor.process_inputs()` 处理靠近 EngineCore 契约的工作：

1. 校验 `SamplingParams` 或 `PoolingParams`，并按实际 Runner 校验 custom logits processors。
2. 校验 LoRA 与 data-parallel rank。
3. 必要时兼容旧的 raw prompt 路径。
4. 拆分 encoder/decoder input。
5. 检查 prompt 长度、token ID 范围和平台限制。token ID 范围由 `BaseRenderer.validate_token_ids()` 判定；
   v0.33.0 起即使 `skip_tokenizer_init=True`（没有 tokenizer）也会按模型词表大小检查，不再整段跳过
   （[47e2b610b](https://github.com/vllm-project/vllm/commit/47e2b610b)，
   [#62911](https://github.com/vllm-project/vllm/pull/62911)）。
6. clone 并补全 request-local 参数，例如默认 `max_tokens` 和 EOS；默认最大输出长度
   按 `max_model_len - prompt_len` 得到后，还要检查 `min_tokens` 是否超过这一余量。
7. 整理多模态 feature。
8. 构造可跨 client 边界传输的 `EngineCoreRequest`。

当前代码已经把 raw prompt 直接传给 `InputProcessor` 标记为 deprecated。推荐理解是：

```text
用户输入 --Renderer--> EngineInput --InputProcessor--> EngineCoreRequest
```

而不是：

```text
用户输入 --InputProcessor 一步完成所有事情--> Request
```

在线 raw prompt 的兼容路径会使用 `process_inputs_async`，它通过 Renderer 的线程池执行
阻塞预处理，避免 tokenization 或媒体处理卡住 asyncio event loop。

[源码] `vllm/v1/engine/input_processor.py` - `InputProcessor.__init__`、
`process_inputs`

```mermaid
flowchart TD
    A[Chat messages / prompt / token IDs] --> B[Renderer]
    B --> C[template + tokenize + multimodal process]
    C --> D[EngineInput]
    D --> E[InputProcessor]
    E --> F{校验}
    F -->|失败| G[VLLMValidationError]
    F -->|通过| H[clone/update params]
    H --> I[EngineCoreRequest]
```

> **读图方法：** 阅读“InputProcessor：从 `EngineInput` 到 `EngineCoreRequest`”这张流程图时，先把方框看成对象或状态，把箭头看成数据或控制的移动。第一遍从上向下建立顺序，第二遍再核对分支发生的条件。

## 2.5 Request ID 为什么有两份

`InputProcessor.assign_request_id()` 会把用户提供的 ID 保存到
`external_req_id`，并通常给内部 `request_id` 追加 8 个随机字符。

```text
external_req_id = "chatcmpl-42"
internal request_id = "chatcmpl-42-a1b2c3d4"
```

这样做不是装饰。多个调用方可能重复使用同一个外部 ID；若内部 Scheduler、KV Cache
和输出状态直接以它为唯一键，就可能互相覆盖。随机化后的 ID 用于内部所有权和资源
管理，外部 ID 用于返回用户、日志关联以及 `abort(internal=False)`。

`OutputProcessor` 维护：

```text
external_req_id -> [internal_req_id, ...]
internal_req_id -> RequestState
```

当 `n > 1` 时，一个父请求还会 fan out 成多个 child request。这说明“一次 HTTP
请求”“一个外部 request ID”“一个 Scheduler request”并不总是一一对应。

在线 `AsyncLLM.add_request` 在本章 revision 下先准备全部 child，按 `n` 检查准入，
然后在第一次跨进程 `await` 前把全部 child 注册到 OutputProcessor，才逐个发送。
这样，即使发送中途被取消，本地仍知道整个父请求包含哪些 child；异常分支用父请求
的内部 ID 执行 `abort` 并重新抛出。这个顺序不是把每个 child 当独立请求重复准入。

[源码] `vllm/v1/engine/async_llm.py` - `AsyncLLM.add_request` 的 fan-out 与异常清理分支

[源码] `vllm/v1/engine/input_processor.py` - `InputProcessor.assign_request_id`

[源码] `vllm/v1/engine/output_processor.py` - `OutputProcessor.add_request`、
`abort_requests`

```mermaid
flowchart LR
    E[external id\nchatcmpl-42] --> I1[internal child 0]
    E --> I2[internal child 1]
    I1 --> S1[RequestState 0]
    I2 --> S2[RequestState 1]
    I1 --> K1[Scheduler/KV state]
    I2 --> K2[Scheduler/KV state]
```

> **读图方法：** 这张图用于压缩“Request ID 为什么有两份”的整体关系。先从左向右找到起点、关键转换和终点，再问每条跨层箭头是否意味着函数调用、消息传递、内存访问或状态更新。

## 2.6 EngineCoreClient：相同接口，不同边界

前端不应根据 EngineCore 是否同进程而重写业务逻辑，因此 V1 用
`EngineCoreClient` 抽象请求的 push/pull。

### `InprocClient`

`InprocClient` 直接构造 `EngineCore`：

- `add_request()` 直接调用 `preprocess_add_request()` 和 `add_request()`。
- `get_output()` 直接调用 `EngineCore.step_fn()`。
- 没有 ZMQ，也没有独立 EngineCore busy loop。
- 用于显式关闭 `VLLM_ENABLE_V1_MULTIPROCESSING` 后的同步兼容或调试路径。当前该环境
  变量默认值为 `1`，所以默认同步 `LLMEngine` 使用的是 `SyncMPClient`，不能把
  `InprocClient` 写成默认离线拓扑。这里的“兼容”也只指 API 门面，不代表回到旧 V0 内核。

### `SyncMPClient`

同步多进程 client 通过 ZMQ 把请求发送到后台 `EngineCoreProc`，并用一个输出线程把
ZMQ 消息放入本地 `outputs_queue`。前端调用 `get_output()` 时阻塞读取这个队列。

### `AsyncMPClient`

异步 client 同样使用 ZMQ，但输出落入 `asyncio.Queue`，`get_output_async()` 可以被
`AsyncLLM.output_handler` await。`add_request_async()` 还会设置 `client_index`，确保多
API Server 场景中输出回到提交请求的 client。

### DP 变体

当 `data_parallel_size > 1` 时：

- external load balancer 路径使用 `DPAsyncMPClient`。
- internal load balancer 路径使用 `DPLBAsyncMPClient`。

它们改变路由和 EngineCore 数量，不改变“前端通过 client 提交
`EngineCoreRequest`、接收 `EngineCoreOutputs`”这一基本契约。v0.33.0 还把 DP wave 的唤醒权交给引擎：
前端不再发送 `FIRST_REQ`，`DPEngineCoreProc.add_request()` 在引擎空闲且未暂停时自己置 `engines_running`，
并经 coordinator 通知其他 rank 开始新一轮 wave
（[a94c31571](https://github.com/vllm-project/vllm/commit/a94c31571)，
[#59559](https://github.com/vllm-project/vllm/pull/59559)）。

[源码] `vllm/v1/engine/core_client.py` - `EngineCoreClient.make_client`、
`make_async_mp_client`、`InprocClient`、`AsyncMPClient`

```mermaid
flowchart TD
    A{asyncio_mode?} -->|否| B{multiprocess_mode?}
    A -->|是| C{multiprocess_mode?}
    B -->|否| D[InprocClient]
    B -->|是| E[SyncMPClient]
    C -->|否| X[NotImplemented]
    C -->|是| F{DP size > 1?}
    F -->|否| G[AsyncMPClient]
    F -->|是, external LB| H[DPAsyncMPClient]
    F -->|是, internal LB| I[DPLBAsyncMPClient]
```

> **读图方法：** 这是“DP 变体”的流程图。先从上向下只追一条主路径，确认输入经过哪些关键阶段到达输出；第二遍再看虚线、回边和旁路，它们通常表示反馈、复用或可选分支。

## 2.7 当前进程拓扑：不要把逻辑对象等同于进程

> **本节先看：** 本节要回答：**当前进程拓扑：不要把逻辑对象等同于进程**。下面的小节会逐层拆开概念、运行过程和源码落点；第一次阅读先抓对象之间的关系，第二次再记类名、字段和分支。

### 单 GPU 在线服务的常见拓扑

在线 `AsyncLLM` 必须使用 multiprocessing client，因此 API 前端与 EngineCore 通常是
两个进程。但当 `world_size == 1` 时，`ParallelConfig` 默认把 executor backend 设为
`"uni"`，`Executor.get_class()` 选择 `UniProcExecutor`。

`UniProcExecutor._init_executor()` 在当前 EngineCore 进程里直接创建
`WorkerWrapperBase`、初始化设备并加载模型。于是常见单 GPU 逻辑是：

```mermaid
flowchart LR
    subgraph P0[进程 0: API Server]
        H[HTTP / asyncio]
        R[OnlineRenderer]
        A[AsyncLLM]
        OP[OutputProcessor]
    end
    subgraph P1[进程 1: EngineCoreProc]
        IT[ZMQ input thread]
        LOOP[EngineCore busy loop]
        OT[ZMQ output thread]
        U[UniProcExecutor]
        W[Worker + Model Runner]
    end
    GPU[(GPU 0)]
    A -->|ZMQ request| IT --> LOOP --> U --> W --> GPU
    GPU --> W --> U --> LOOP --> OT -->|ZMQ output| OP
```

> **读图方法：** 阅读“单 GPU 在线服务的常见拓扑”这张流程图时，先把方框看成对象或状态，把箭头看成数据或控制的移动。第一遍从左向右建立顺序，第二遍再核对分支发生的条件。

这里有一个对源码阅读非常重要的修正：

> “每个 GPU 都有一个逻辑 Worker”是成立的；“每个 GPU 必然有一个额外的独立 Worker
> 进程”在当前 revision 并不成立。单 rank 的 `UniProcExecutor` 把 Worker 放在
> EngineCore 进程内。

官方 `docs/design/arch_overview.md` 的进程计数把 Worker 描述为独立进程，适合作为
多进程部署的总体直觉，但阅读当前代码时必须结合 executor backend 修正。

[源码] `vllm/config/parallel.py` - `ParallelConfig.__post_init__`

[源码] `vllm/v1/executor/abstract.py` - `Executor.get_class`

[源码] `vllm/v1/executor/uniproc_executor.py` - `UniProcExecutor._init_executor`

### TP/PP 多 rank 路径

当 backend 为 `mp` 时，`MultiprocExecutor` 创建多个 Worker 进程，并通过共享内存
message queue 等机制广播 `SchedulerOutput`、收集执行结果。此时才更接近：

```text
API process
  + EngineCore process
  + W_engine worker processes
```

这里的 worker 进程数就是这个引擎的 world size：

$$
W_{\mathrm{engine}}=\mathrm{TP}\times\mathrm{PP}\times\mathrm{PCP}
$$

**白话：**一个 DP rank 里的模型被切成 TP、PP、PCP 三个方向，每个切片落在一个 worker 进程上，所以进程数是三个方向路数的乘积。

**其中：**$\mathrm{TP}$、$\mathrm{PP}$、$\mathrm{PCP}$ 分别是 tensor parallel、pipeline parallel、prefill context parallel 的路数，都是无单位正整数；
$W_{\mathrm{engine}}$ 是一个 DP rank 的 world size（个 worker）；$W_{\mathrm{all\_dp}}=W_{\mathrm{engine}}\times\mathrm{DP}$ 是所有 DP rank 的 worker 总数（本书记号，见符号表 §1.5）；两者都不含 API 进程和 EngineCore 进程；DCP 不增加进程，所以不在乘积里。

**怎么算出来：**三个方向相互独立：先沿 TP 切成 $\mathrm{TP}$ 份，每一份再沿 PP 切成 $\mathrm{PP}$ 个 stage，每个 stage 再沿 PCP 切成 $\mathrm{PCP}$ 份，所以总份数是连乘，每份一个 rank、一个 worker 进程。
算例（教学数字）：$\mathrm{TP}=4$、$\mathrm{PP}=2$、$\mathrm{PCP}=1$ 时 $W_{\mathrm{engine}}=4\times2\times1=8$，
单节点上这个引擎就是 1 个 EngineCore 进程加 8 个 worker 进程；若还有 $\mathrm{DP}=2$，就是 2 个这样的引擎（各有自己的 EngineCore 和 executor），
全部 worker 是 $W_{\mathrm{all\_dp}}=W_{\mathrm{engine}}\times\mathrm{DP}=16$ 个。多节点时一个 DP rank 的 worker 分布在 `nnodes_within_dp` 个节点上，每个节点只起 $W_{\mathrm{engine}}/\mathrm{nnodes\_within\_dp}$ 个（`local_world_size`）。

**核对：**`ParallelConfig.__post_init__` 把 `world_size` 赋为 `pipeline_parallel_size * tensor_parallel_size * prefill_context_parallel_size`
（`vllm/config/parallel.py:1062-1066`）；`MultiprocExecutor._init_executor` 断言 `world_size == tp_size * pp_size * pcp_size`
（`vllm/v1/executor/multiproc_executor.py:164`），每节点 worker 数是 `local_world_size = world_size // nnodes_within_dp`（`vllm/config/parallel.py:936-937`）。
`external_launcher` 路径会把 DP 并入 `world_size`，这个特例和完整的 rank 推导放在第 09 章 §9.3.1。

### 离线 in-process 路径

显式设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 时，离线 `LLMEngine` 使用
`InprocClient`。配合单 rank 的 `UniProcExecutor`，此时 EngineCore 和 `UniProcExecutor`
Worker 都在调用者进程，`LLM.generate()` 的调用线程主动推进 step：

```mermaid
flowchart LR
    subgraph ONE[同一 Python 进程]
        L[LLM.generate] --> LE[LLMEngine]
        LE --> IC[InprocClient]
        IC --> EC[EngineCore]
        EC --> UE[UniProcExecutor]
        UE --> W[Worker / Model Runner]
    end
    W <--> G[(GPU)]
```

> **读图方法：** 这张图用于压缩“离线 in-process 路径”的整体关系。先从左向右找到起点、关键转换和终点，再问每条跨层箭头是否意味着函数调用、消息传递、内存访问或状态更新。

因此，任何进程图都必须带上三个条件：入口、EngineCore client 类型、executor backend。

## 2.8 在线请求怎样跨过 ZMQ 边界

以 `AsyncMPClient` 和 `EngineCoreProc` 为主线：

1. `AsyncLLM._add_request()` 先在前端 `OutputProcessor` 注册 `RequestState`。
2. 然后 await `engine_core.add_request_async(request)`。
3. client 用 msgpack 编码 `EngineCoreRequest`，通过 ZMQ multipart 发送。
4. EngineCore 的 input socket thread 收到消息并反序列化。
5. input thread 调用 `preprocess_add_request()`，把 wire object 变成内部 `Request`。
6. 处理后的 `(request_type, request)` 进入线程安全的 `input_queue`。
7. busy loop 从队列取出 ADD，调用 `EngineCore.add_request()`。
8. `EngineCore.add_request()` 最终交给 `Scheduler.add_request()`。

为什么先在前端注册、再做第一次 await？源码注释给出了并发理由：并发 task 必须在
请求跨进程前就能看见该请求，避免输出或取消到来时本地没有对应状态。

```mermaid
sequenceDiagram
    participant G as AsyncLLM.generate
    participant OP as OutputProcessor
    participant C as AsyncMPClient
    participant Z as ZMQ
    participant IT as EngineCore input thread
    participant Q as input_queue
    participant EC as EngineCore busy loop
    participant S as Scheduler

    G->>OP: add_request(RequestState)
    G->>C: add_request_async(EngineCoreRequest)
    C->>Z: msgpack multipart
    Z->>IT: receive + decode
    IT->>IT: preprocess_add_request -> Request
    IT->>Q: put(ADD, Request)
    EC->>Q: get()
    EC->>S: add_request(Request)
```

> **读图方法：** 这是“在线请求怎样跨过 ZMQ 边界”的时序图。先从左到右确认参与者分别负责什么，再从上到下追踪消息；第一遍只看正常路径，第二遍再看返回、异步消息和失败分支。

### 为什么在 input thread 中构造 `Request`

`preprocess_add_request()` 可能读取多模态 receiver cache、计算 block hash、启动
structured-output grammar 初始化。把它放在 input thread，可以与 GPU forward 并行，
减少 busy loop 的串行 CPU 开销。

这也说明 input thread 不只是“搬运 socket 字节”，它承担一部分可并行的 request
初始化。但真正改变 Scheduler 队列仍发生在 busy loop，避免多个线程任意修改核心
状态。

input thread 里的初始化也可能失败。v0.33.0 起，`preprocess_add_request()` 抛出
`VLLMValidationError`（目前用于“请求使用的 structured-output backend 与引擎已锁定的不同”）时，
线程调用 `_handle_request_validation_error()`（`vllm/v1/engine/core.py:2413`）：不让该请求进入 Scheduler，
而是回送一个 `finish_reason=ERROR`、`RequestError(code="validation_error")` 的输出，前端据此抛回客户端错误
（[88077b746](https://github.com/vllm-project/vllm/commit/88077b746)，
[#49625](https://github.com/vllm-project/vllm/pull/49625)）。其它异常仍走原来的
`_handle_request_preproc_error()`。

多模态缓存失配也在这条链路上：EngineCore 发现 receiver cache 缺项时，输出会带 `mm_cache_miss_hashes`。
v0.33.0 的 `AsyncLLM` 输出循环在调用 `process_outputs()` **之前**，先通过
`renderer.queue_mm_cache_invalidations()`（`vllm/v1/engine/async_llm.py:916`）把这些 hash 排进渲染器的待失效集合；
下一次渲染在查缓存前由 `_drain_mm_cache_invalidations()` 统一清掉（`vllm/renderers/base.py:397`、`:408`）。
这样排队中的渲染也不会在客户端重试前再次命中已失效的 sender cache
（[70806b631](https://github.com/vllm-project/vllm/commit/70806b631)，
[#62164](https://github.com/vllm-project/vllm/pull/62164)）。

[源码] `vllm/v1/engine/core.py` - `EngineCore.preprocess_add_request`、
`EngineCoreProc.process_input_sockets`、`_handle_client_request`

## 2.9 一次 EngineCore step 的骨架

当 Scheduler 有请求时，`EngineCore.step()` 的主线非常短，却串起整套系统：

```text
if no scheduler requests:
    return {}, False

scheduler_output = scheduler.schedule(...)
future = model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = model_executor.sample_tokens(grammar_output)

process aborts received during model execution
engine_outputs = scheduler.update_from_output(scheduler_output, model_output)
return engine_outputs, model_was_executed
```

它可以拆成五个阶段：

| 阶段 | 所有者 | 输入 | 输出 |
|---|---|---|---|
| 选择工作 | Scheduler | waiting/running 请求、token/KV 预算 | `SchedulerOutput` |
| 执行模型 | Executor/Worker | `SchedulerOutput` | future / `ModelRunnerOutput` |
| 必要时采样 | Worker/Sampler | logits、grammar mask | sampled token |
| 处理并发 abort | EngineCore | abort queue | 被终止的 Scheduler request |
| 更新状态 | Scheduler | 调度快照 + 模型输出 | `EngineCoreOutputs` |

```mermaid
sequenceDiagram
    participant EC as EngineCore
    participant S as Scheduler
    participant EX as Executor
    participant W as Worker / ModelRunner
    participant AQ as aborts_queue

    EC->>S: schedule()
    S-->>EC: SchedulerOutput
    EC->>EX: execute_model(non_block=True)
    EX->>W: execute_model
    W-->>EX: ModelRunnerOutput/future
    EX-->>EC: future.result(): output 或 None
    opt output 为 None，需要单独采样
        EC->>EX: sample_tokens(grammar_output)
        EX->>W: sample_tokens
        W-->>EC: ModelRunnerOutput 经 Executor 返回
    end
    EC->>AQ: drain aborts
    EC->>S: update_from_output(...)
    S-->>EC: EngineCoreOutputs
```

> **读图方法：** 这是“一次 EngineCore step 的骨架”的时序图。先从左到右确认参与者分别负责什么，再从上到下追踪消息；第一遍只看正常路径，第二遍再看返回、异步消息和失败分支。

### 为什么 abort 要在 `update_from_output()` 前处理

假设一个请求的最后一步正在 GPU 上执行，此时客户端取消。如果先让
`update_from_output()` 把它标成 `FINISHED_LENGTH_CAPPED`，再处理 abort，KV connector
等资源清理逻辑可能看到错误的完成原因。当前代码在模型结果返回后、Scheduler 消费
结果前调用 `_process_aborts_queue()`，让取消优先。

`tests/v1/engine/test_abort_final_step.py::test_abort_during_final_step` 专门构造了这一竞争
条件，验证 connector 观察到 `FINISHED_ABORTED`。

[源码] `vllm/v1/engine/core.py` - `EngineCore.step`、`_process_aborts_queue`

[测试] `tests/v1/engine/test_abort_final_step.py` -
`test_abort_during_final_step`

## 2.10 数据形态怎样逐层变化

同一个请求在不同层不是同一个 Python 对象。每次转换都在缩小或改变责任边界。

```mermaid
flowchart TD
    A[HTTP JSON / Python Prompt] --> B[EngineInput]
    B --> C[EngineCoreRequest]
    C --> D[Request]
    D --> E[SchedulerOutput]
    E --> F[GPU input tensors]
    F --> G[ModelRunnerOutput]
    G --> H[EngineCoreOutput]
    H --> I[RequestOutput]
    I --> J[SSE JSON / Python return]
```

> **读图方法：** 这张图用于压缩“数据形态怎样逐层变化”的整体关系。先从上向下找到起点、关键转换和终点，再问每条跨层箭头是否意味着函数调用、消息传递、内存访问或状态更新。

### `EngineInput`

Renderer 的输出，已经完成模板化和 tokenization。它可能表示纯 token、prompt embeds、
多模态输入或 encoder-decoder 输入。

### `EngineCoreRequest`

前端和 EngineCore 之间的消息契约。关键字段包括：

- 内部/外部 request ID
- prompt token IDs 或 embeds
- `SamplingParams` / `PoolingParams`
- arrival time、priority、LoRA
- client index、DP rank
- 多模态 feature、trace header、session 信息
- 可选 `kv_hints`：随请求传入后端的 KV 提示，不等于已经命中或拥有 cache blocks

它使用 `msgspec.Struct`，适合 msgpack 序列化。它不是 Scheduler 的完整运行状态。

### `Request`

EngineCore 内部的可变状态对象。它增加：

- `RequestStatus`
- `num_computed_tokens`
- output token 列表
- speculative token、in-flight token、preemption 计数
- block hashes、prefill stats、streaming state

`EngineCoreRequest` 是“提交消息”，`Request` 是“被系统持续推进的实体”。

### `SchedulerOutput`

一次 step 的工作说明，包含：

- 新请求的完整数据 `scheduled_new_reqs`
- 已缓存请求的增量数据 `scheduled_cached_reqs`
- 每请求本步 token 数
- 新 block IDs、finished/preempted request IDs
- encoder input、spec decode token、common prefix 信息

它回答“这一轮 Worker 应该做什么”，不等于长期 Request 状态。它是普通可变
`dataclass`，不是 Python 层面的不可变对象；异步执行时还会更新草稿 token 等字段。

[源码] `vllm/v1/core/sched/output.py` - `SchedulerOutput`

### `ModelRunnerOutput`

执行侧返回的模型级结果，主要包含采样 token、logprobs、pooling output、KV/encoder
connector metadata、draft token 等。它还不是用户文本。

### `EngineCoreOutputs` 与 `EngineCoreOutput`

两者分别表示输出容器与单请求输出，不能只靠名称中的复数判断进程边界：

- `EngineCoreOutputs` 按 client 组织本轮输出；`outputs` 列表只包含本轮需要交付的请求项，
  不要求该 client 的所有请求都出现。它还可以携带统计或可选的 `utility_output`。
  多进程 client 通过 ZMQ 收到此对象，同进程 client 也使用同一输出类型。
- `EngineCoreOutput` 是 `outputs` 中的单请求输出单元，包含 `new_token_ids`、
  finish reason、stop reason、事件及 prefill stats 等。

[源码] `vllm/v1/engine/__init__.py` - `EngineCoreOutputs`、`EngineCoreOutput`

[源码] `vllm/v1/engine/core_client.py` - `InprocClient.get_output`

### `RequestOutput`

OutputProcessor 增量 detokenize、处理 stop string、logprobs 和多路 sampling 后形成的用户
层结果。在线路径把它送入每请求 collector；离线路径把它放入返回列表。

| 数据 | 主要所有者 | 是否跨 ZMQ | 是否含文本 | 生命周期 |
|---|---|---:|---:|---|
| `EngineInput` | Renderer/frontend | 通常不直接跨 | 可保留 prompt | 单次预处理 |
| `EngineCoreRequest` | InputProcessor/client | 是，MP 路径 | 通常以 token 为主 | 提交消息 |
| `Request` | EngineCore/Scheduler | 否 | 否 | 整个推理请求 |
| `SchedulerOutput` | Scheduler/Worker | Executor 决定 | 否 | 一个 step |
| `ModelRunnerOutput` | Worker/EngineCore | Executor 决定 | 否 | 一个 step |
| `EngineCoreOutput` | Scheduler/frontend | 是，MP 路径 | 否 | 一个 step/请求 |
| `RequestOutput` | OutputProcessor/API | 否 | 是 | 增量或最终输出 |

## 2.11 OutputProcessor 怎样把 token 变回用户输出

EngineCore 返回 token ID 后，前端还要完成多项工作：

1. 根据内部 request ID 找到 `RequestState`。
2. 更新统计信息和 prefill cache 命中数据。
3. 用 `IncrementalDetokenizer` 把新 token 拼成文本。
4. 在文本层检查 stop string。
5. 整理 prompt/sample logprobs。
6. 构造 `CompletionOutput` 和 `RequestOutput`。
7. 如果完成，删除前端 request state。
8. 若 stop string 由前端发现而 EngineCore 尚未完成，把 request ID 放入
   `reqs_to_abort`，通知后端清理。

**v0.33.0 新增的两个分支。** `OutputProcessor.process_outputs()` 在做上面 2～8 步之前，
先看 `EngineCoreOutput.error`（`vllm/v1/engine/output_processor.py:767`）：若带错误且请求有 collector，
说明请求在**到达 EngineCore 时就被拒绝**（例如引擎暂停，见 2.14），模型一个 token 都没跑过；
此时直接把 `error.to_exception()` 放进 collector 并 `_finish_request()`，不再统计、detokenize。
第二个分支是流式输入（streaming input）会话：一个会话由多个输入 chunk 组成，每个 chunk 结束时
`finish_reason` 非空但会话并未结束。`streaming_chunk_boundary`（`:847`）只在 `finish_reason` 不是
`ABORT`/`ERROR` 时为真，它让该输出的 `finished` 保持 `False`，并切换到下一个 chunk；
遇到 `ABORT`/`ERROR` 则当作终态，清理并通知 EngineCore，防止“僵尸会话”
（[146350155](https://github.com/vllm-project/vllm/commit/146350155)，
[#58942](https://github.com/vllm-project/vllm/pull/58942)）。

```mermaid
flowchart TD
    IN["一条 EngineCoreOutput"] --> FIND{"前端 RequestState 还在吗?"}
    FIND -->|"否: 已 abort"| SKIP["忽略该输出"]
    FIND -->|"是"| ERR{"output.error 非空且有 collector?"}
    ERR -->|"是: 到达即被拒绝"| REJ["collector.put(error.to_exception()); _finish_request"]
    REJ --> SI1{"流式输入会话?"}
    SI1 -->|"是"| AB1["加入 reqs_to_abort, 防止后续输入 chunk 重建会话"]
    ERR -->|"否"| NORM["统计 -> detokenize -> stop string -> logprobs"]
    NORM --> MK["make_request_output -> collector.put"]
    MK --> FIN{"finish_reason 非空?"}
    FIN -->|"否"| NEXT["等待下一个 token"]
    FIN -->|"是"| SI2{"streaming_chunk_boundary?"}
    SI2 -->|"是: 流式输入且不是 ABORT/ERROR"| CHUNK["finished 置 False, 切换到下一个输入 chunk"]
    SI2 -->|"否"| DONE["_finish_request; 流式输入或后端未结束时加入 reqs_to_abort"]
```

> **读图方法：** 这是 `process_outputs` 对单条输出的分支图。先从上往下看三道关卡：状态是否存在、是否带错误、是否结束；右侧两个叶子（拒绝清理、正常结束）都会释放前端状态，而“切换到下一个输入 chunk”是唯一不释放状态的结束分支。

其他实现细节：`n>1` 的并行采样里，每个 child 结束时会从 parent 的 `child_requests` 集合中移除自己
（`:932`），最后一个 child 结束才释放 parent；池化（pooling）输出的构造被提到
`stream_interval` 逻辑之前，使 pooling 请求不再受 token stream interval 影响
（[f0c50148a](https://github.com/vllm-project/vllm/commit/f0c50148a)，
[#55226](https://github.com/vllm-project/vllm/pull/55226)）。

### 离线输出

离线 `LLMEngine.step()` 调用 `engine_core.get_output()`，然后
`output_processor.process_outputs()`。因为 `RequestState.queue is None`，结果被收集到
`request_outputs` 列表并返回调用线程。

### 在线输出

在线 `AsyncLLM` 有一个长期运行的 `output_handler` task：

```text
await EngineCoreClient.get_output_async()
  -> chunk EngineCoreOutputs to avoid blocking event loop too long
  -> OutputProcessor.process_outputs()
  -> RequestOutputCollector.put()
  -> each AsyncLLM.generate() task reads its own collector
```

`RequestOutputCollector` 不是普通无限队列。它只有一个当前 output slot；当 DELTA 模式
下 producer 比 consumer 快时，后续 delta 会合并到已有 `RequestOutput`，减少积压，且
保留 `n > 1` 的不同 choice index。

流式输入请求改用子类 `StreamingInputOutputCollector`（`vllm/v1/engine/output_processor.py:129`）：
当前 slot 里已经是“chunk 结束”输出（`_is_chunk_boundary`，`:117`：`finish_reason` 非空而请求未结束）时，
后到的输出不再合并进去，而是进入 `_backlog` 队列，`get()` 取走一条后再把队首补进 slot。
这样一个 chunk 的结束事件不会被下一个 chunk 的 delta 吞掉；收到异常则清空 backlog
（[957d512e9](https://github.com/vllm-project/vllm/commit/957d512e9)，
[#58941](https://github.com/vllm-project/vllm/pull/58941)）。

[源码] `vllm/v1/engine/output_processor.py` - `RequestOutputCollector`、
`StreamingInputOutputCollector`、`OutputProcessor.process_outputs`

[测试] `tests/v1/engine/test_output_processor.py` - collector merge、`test_abort_requests`，以及 v0.33.0 新增的
`test_streaming_input_collector_preserves_chunk_boundary`、`test_streaming_input_stop_is_chunk_boundary`、
`test_streaming_input_abort_error_is_terminal`、`test_request_rejected_by_engine`

```mermaid
sequenceDiagram
    participant OT as EngineCore output thread
    participant C as AsyncMPClient
    participant H as AsyncLLM output_handler
    participant OP as OutputProcessor
    participant Q as RequestOutputCollector
    participant G as generate task
    participant API as SSE formatter

    OT->>C: EngineCoreOutputs over ZMQ
    C->>H: asyncio.Queue
    H->>OP: process_outputs(slice)
    OP->>Q: put(RequestOutput)
    G->>Q: get_nowait() or await get()
    G-->>API: yield RequestOutput
    API-->>API: format data: JSON
```

> **读图方法：** 这是“在线输出”的时序图。先从左到右确认参与者分别负责什么，再从上到下追踪消息；第一遍只看正常路径，第二遍再看返回、异步消息和失败分支。

## 2.12 从请求到输出的完整在线时序

下面把前面所有局部拼成一条最小 happy path。为了保持清晰，图中省略 DP、multi-modal
cache 和 speculative decoding。

```mermaid
sequenceDiagram
    actor User
    participant API as OpenAIServingChat
    participant R as OnlineRenderer
    participant A as AsyncLLM
    participant IP as InputProcessor
    participant C as AsyncMPClient
    participant EC as EngineCoreProc
    participant S as Scheduler
    participant W as Executor/Worker
    participant OP as OutputProcessor

    User->>API: POST chat messages
    API->>R: render_chat(request)
    R-->>API: conversation + EngineInput
    API->>A: generate(EngineInput, SamplingParams)
    A->>IP: process_inputs
    IP-->>A: EngineCoreRequest
    A->>OP: register RequestState
    A->>C: add_request_async
    C->>EC: ZMQ ADD
    EC->>S: add_request

    loop until finished
        EC->>S: schedule
        S-->>EC: SchedulerOutput
        EC->>W: execute_model
        W-->>EC: output 或 None
        opt output 为 None
            EC->>W: sample_tokens(grammar_output)
            W-->>EC: ModelRunnerOutput
        end
        EC->>S: update_from_output
        S-->>EC: EngineCoreOutput
        EC->>C: ZMQ output
        C-->>A: get_output_async
        A->>OP: process_outputs
        OP-->>A: per-request collector
        A-->>API: yield RequestOutput
        API-->>User: SSE chunk
    end
```

> **读图方法：** 这是“从请求到输出的完整在线时序”的时序图。先从左到右确认参与者分别负责什么，再从上到下追踪消息；第一遍只看正常路径，第二遍再看返回、异步消息和失败分支。

图中以 `EngineCore.step` 的非队列路径展开执行与采样；启用异步调度等配置时会选择
`step_with_batch_queue`，第 03、08 章再解释多批次在途。不要由本图推断每次提交后
CPU 都必须等到 GPU 完成才允许下一次调度。

这里最容易忽略的是：一次用户请求通常经历许多次 loop。只有 prefill 和若干次 decode
全部完成，或者遇到 EOS、stop、length、abort、error 等结束条件，生命周期才终止。

## 2.13 Busy loop、线程与队列

`EngineCoreProc` 为 ZMQ 包装器。它在初始化时创建：

- input socket thread：Socket -> decode/preprocess -> `input_queue`
- output socket thread：`output_queue` -> encode -> Socket
- 主线程 busy loop：处理 client request、执行 EngineCore step

输入/输出线程的意义不只是代码整洁。ZMQ I/O 会释放 GIL，序列化/反序列化也可与 GPU
执行发生部分重叠，从而减少主循环等待网络消息的时间。

busy loop 的骨架是：

```text
while shutdown state allows running:
    process input queue until work exists
    maybe publish DP load stats
    run one engine step
    maybe publish DP load stats again
```

当没有 work 时，`_process_input_queue()` 可以阻塞等待输入；有 work 后，仍先用
`get_nowait()` 处理队列中已到达的 ADD、ABORT、UTILITY，再推进 step。不能把忙碌
理解为停止接单，否则无法解释 continuous batching 和运行中的取消。
若本 step 没有实际模型执行，但 Scheduler 仍在等待远端 KV 等后台工作，代码会短暂
sleep 1ms，让后台 transfer thread 获得进展机会。

v0.33.0 给 busy loop 加了看门狗：`EngineCoreProc.run_engine_core` 在进入 `run_busy_loop()` 前启动
`start_watch_dog(f"engine_{dp_rank}", …)`（`vllm/v1/engine/core.py:1757`），每次
`_process_engine_step()` 开头 `get_watch_dog().feed()`（`:1872`）；超时未喂时看门狗转储所有线程栈用于诊断。
阻塞读取 `input_queue` 时包在 `get_watch_dog().disable()` 里（接口注释：暂停超时检测，例如阻塞等待空闲输入）
（[1c8799197](https://github.com/vllm-project/vllm/commit/1c8799197)，
[#55700](https://github.com/vllm-project/vllm/pull/55700)）。

[源码] `vllm/v1/engine/core.py` - `EngineCoreProc.run_busy_loop`、
`_process_input_queue`、`_process_engine_step`

```mermaid
flowchart TD
    A[busy loop] --> B{shutdown?}
    B -->|继续| C{has_work?}
    C -->|否| D[阻塞读取 input_queue]
    D --> E[handle ADD/ABORT/UTILITY]
    E --> C
    C -->|是| DRAIN[非阻塞排空已有 input_queue 消息]
    DRAIN --> F[step_fn]
    F --> G[outputs -> output_queue]
    G --> H[post_step]
    H --> A
    B -->|退出| I[shutdown resources]
```

> **读图方法：** 这张图用于压缩“Busy loop、线程与队列”的整体关系。先从上向下找到起点、关键转换和终点，再问每条跨层箭头是否意味着函数调用、消息传递、内存访问或状态更新。

## 2.14 取消、stop string 与异常传播

> **本节先看：** 本节要回答：**取消、stop string 与异常传播**。下面的小节会逐层拆开概念、运行过程和源码落点；第一次阅读先抓对象之间的关系，第二次再记类名、字段和分支。

### 客户端断开

在线调用者断开连接时，处理 HTTP 的 task 或 `AsyncLLM.generate()` generator 会被
取消。`generate()` 捕获 `asyncio.CancelledError` / `GeneratorExit` 后调用：

```text
AsyncLLM.abort(internal=True)
  -> OutputProcessor.abort_requests
  -> EngineCoreClient.abort_requests_async
  -> ZMQ ABORT
  -> EngineCore.abort_requests
  -> Scheduler.finish_requests(FINISHED_ABORTED)
```

前端清理和后端清理缺一不可：

- 只清前端：GPU/KV 仍可能继续为无人消费的请求工作。
- 只清后端：前端 collector、detokenizer 和 ID mapping 仍可能泄漏。

### stop string 由前端发现

token 级 EOS 可以由 EngineCore 处理，但任意字符串 stop 依赖增量 detokenize 后的文本。
若 `OutputProcessor` 发现 stop string，而 `EngineCoreOutput.finished` 仍为 false，它先向
用户产生完成输出，再把内部 ID 放进 `reqs_to_abort`，由 Engine client 通知 Scheduler
清理剩余状态。v0.33.0 起，流式输入会话在前端终结时也会加入 `reqs_to_abort`：
传输中的下一个输入 chunk 可能让 EngineCore 重建同一个会话，必须一并清掉。

### 暂停期间到达的新请求会被拒绝

`AsyncLLM.pause_generation(mode, clear_cache)` 用于权重更新等需要停住生成的场景。v0.32.0 里新请求在暂停期间
只是不被调度；v0.33.0 区分了模式：`abort` 与 `wait` 模式下，EngineCore 在暂停开始到
`resume_generation` 之间**拒绝**新请求，`keep` 模式仍然接收并排队。实现分三步：

1. `AsyncLLM.pause_generation()` 现在调用 `engine_core.pause_generation_async()`（utility `pause_generation`），
   `EngineCoreProc.pause_generation()`（`vllm/v1/engine/core.py:2471`）在原 `pause_scheduler` 之外设置
   `reject_new_requests = (mode != "keep")`；`resume_scheduler()` 把它清回 `False`。睡眠等其他暂停不改这个标志。
2. busy loop 处理 ADD 时先过 `_reject_add_while_paused()`（`:2017`）：已存在于 Scheduler 的会话输入、
   `abort_immediately` 的 KV 清理标记不算新请求，放行；其余请求回送
   `finish_reason=ERROR`、`RequestError(code="engine_paused", retryable=True)` 的输出，
   带 `kv_transfer_params` 的请求还会以 `abort_immediately` 重新加入，让 KV connector 释放远端块。
3. 前端 `process_outputs()` 的 error 分支（见 2.11）把它变成 `RetryableRequestError`
   （`vllm/exceptions.py:139`，`status_code` 为 503）抛给调用方；`AsyncLLM` 把 `GenerationError` 与
   `VLLMClientError` 归为同一类请求级失败，并对 `n>1` 的并行采样 `abort` 其余 child，避免一半请求被拒、
   另一半继续跑（[d5b36699b](https://github.com/vllm-project/vllm/commit/d5b36699b)，
   [#51488](https://github.com/vllm-project/vllm/pull/51488)）。

```mermaid
sequenceDiagram
    participant U as 调用方(训练或运维)
    participant A as AsyncLLM
    participant EC as EngineCoreProc
    participant OP as OutputProcessor
    participant G as 新请求的 generate task
    U->>A: pause_generation(mode)
    A->>EC: utility pause_generation(mode, clear_cache)
    Note over EC: abort 与 wait 模式 reject_new_requests=True，keep 模式为 False
    G->>A: 新请求
    A->>EC: ADD
    EC->>EC: _reject_add_while_paused
    EC-->>OP: EngineCoreOutput(finish_reason=ERROR, error=engine_paused)
    OP-->>G: collector.put(RetryableRequestError)
    U->>A: resume_generation
    A->>EC: resume_scheduler 后 reject_new_requests=False
```

> **读图方法：** 这是暂停期间准入判定的时序图。先看上半段 `pause_generation` 如何在 EngineCore 里设置拒绝标志，再看中间新请求被 `_reject_add_while_paused` 拦下并以带错误的输出原路返回，最后一行表示恢复后标志清除、请求重新被接收。

[源码] `vllm/v1/engine/core.py` - `EngineCoreProc.pause_generation`、`_reject_add_while_paused`

[源码] `vllm/outputs.py` - `RequestError`

[测试] `tests/v1/engine/test_async_llm.py` - `test_pause_admission_policy_per_mode`、
`test_pause_mid_fanout_rejects_the_whole_request`、`test_retryable_request_error_reaches_caller`

### EngineCore 或 output handler 出错

`AsyncLLM.output_handler` 捕获异常后调用 `OutputProcessor.propagate_error()`，把异常写入
所有活跃 request collector。每个 `generate()` task 在读取 collector 时重新抛出，避免
请求永久挂起。

```mermaid
sequenceDiagram
    actor User
    participant G as generate task
    participant OP as OutputProcessor
    participant C as EngineCoreClient
    participant EC as EngineCore
    participant S as Scheduler

    User -x G: disconnect / cancel
    G->>OP: abort_requests(internal id)
    OP-->>G: final abort output + ids to clean
    G->>C: abort_requests_async(ids)
    C->>EC: ZMQ ABORT
    EC->>S: finish_requests(FINISHED_ABORTED)
    S-->>EC: free request/KV ownership
```

> **读图方法：** 这是“EngineCore 或 output handler 出错”的时序图。先从左到右确认参与者分别负责什么，再从上到下追踪消息；第一遍只看正常路径，第二遍再看返回、异步消息和失败分支。

[源码] `vllm/v1/engine/async_llm.py` - `AsyncLLM.generate`、`abort`、
`_run_output_handler`

[源码] `vllm/v1/engine/output_processor.py` - `abort_requests`、
`propagate_error`

[测试] `tests/v1/engine/test_async_llm.py` - `test_abort`

## 2.15 背压和队列不是同一回事

在线系统里至少有几种不同含义的“队列”：

| 队列/集合 | 所在层 | 作用 |
|---|---|---|
| API 并发 task | API Server | 等待/处理 HTTP 请求 |
| `RequestOutputCollector` | 每请求前端 | 交付增量输出，并可合并 delta |
| client `outputs_queue` | EngineCoreClient | 接收跨进程 EngineCoreOutputs |
| EngineCore `input_queue` | EngineCoreProc | Socket thread 向 busy loop 交接消息 |
| EngineCore `output_queue` | EngineCoreProc | busy loop 向 output thread 交接结果 |
| Scheduler waiting/running | EngineCore | 决定请求何时获得 token/KV 预算 |

下图把这几处放到同一张拓扑里，回答一个问题：请求和输出在哪里排队、哪一处有容量上限、哪一处会拒绝、
哪一处只是积压。

```mermaid
flowchart TB
    subgraph API["API Server 进程 (asyncio)"]
        H["generate 并发 task<br/>HTTP 请求"] --> ADM{"check_admission<br/>唯一的容量拒绝点"}
        ADM -- "超限" --> REJ["QueueOverflowError<br/>MaxQueuedTokensError"]
        ADM -- "通过" --> OP["OutputProcessor<br/>request_states"]
        OP --> CL["AsyncMPClient<br/>add_request_async"]
        CQ["client outputs_queue<br/>无 maxsize"] --> OH["output_handler task"]
        OH --> OP
        OP -- "put 并入" --> COL["RequestOutputCollector<br/>每请求单槽"]
        COL --> H
    end
    subgraph CORE["EngineCore 进程"]
        IT["input socket thread"] --> IQ["input_queue<br/>无 maxsize"]
        IQ --> BL["busy loop<br/>step_fn"]
        BL <--> SCH["Scheduler<br/>waiting / running"]
        BL --> OQ["output_queue<br/>无 maxsize"]
        OQ --> OT["output socket thread"]
        BL -- "SchedulerOutput" --> RB["Executor 到 Worker<br/>ShmRingBuffer 10 槽"]
    end
    CL -. "ZMQ ROUTER/DEALER<br/>SNDHWM=0" .-> IT
    OT -. "ZMQ PUSH/PULL<br/>HWM=0" .-> CQ
    OP -. "读请求数与排队 token 数" .-> ADM
    classDef gate fill:#ffe9e9,stroke:#c0392b;
    classDef bounded fill:#fff6dd,stroke:#b9770e;
    classDef unbounded fill:#e8f4fd,stroke:#2e86c1;
    class ADM,REJ gate;
    class SCH,RB,COL bounded;
    class CQ,IQ,OQ unbounded;
```

> **读图方法：** 先沿实线走一遍“请求入口”：顶部的 `generate` task 进入 `check_admission`，超限就在这里被拒绝，通过后才登记
> `OutputProcessor`、经虚线（跨进程 ZMQ）进入 EngineCore 的 `input_queue`，由 busy loop 交给 Scheduler。
> 再沿回程看输出：左侧 EngineCore 里的 `output_queue` → ZMQ → 右下 client 的 `outputs_queue` → `output_handler` → 每请求的 `RequestOutputCollector` → 回到顶部的 `generate`。
> 颜色只表示容量性质：红色是拒绝点；蓝色是没有容量上限的队列（只会积压，不会拒绝）；黄色是各自有约束的位置（Scheduler 由 token/KV 预算限制谁前进，
> Collector 是单槽并入而不是队列，ShmRingBuffer 写满时写端等待读端）。从 OP 指向 check_admission 的点线表示：排队 token 数读的是前端 OutputProcessor 里的状态；请求数在没有共享计数时也读这里（有 mp_admission_counters 时读共享计数）。两者都不是 EngineCore 里的 waiting 数。

图上的几个事实都能在固定 commit 里找到：

- `check_admission` 在 `n=1` 时由 `AsyncLLM._add_request` 在登记 `OutputProcessor` 之前调用（`vllm/v1/engine/async_llm.py:592`），
  `n>1` 时由 `AsyncLLM.add_request` 的扇出路径在登记所有 child 之前调用（`:565`）；容量超限抛出的 `QueueOverflowError`、`MaxQueuedTokensError` 对应 HTTP 503（docstring 见 `:327-354`）。
  开启 fault tolerance 且引擎不健康时，同一个函数还会先抛 `EngineFaultedError`（`:355-364`），所以图上标的是“唯一的容量拒绝点”，不是唯一会抛错的位置。
- 本地回退用的请求数是 `OutputProcessor.request_states` 的个数（`vllm/v1/engine/output_processor.py:576-577`），排队 token 数是其中仍在 prefill 的请求的 `prompt_len` 之和（`:582-592`）。
- `input_queue` 与 `output_queue` 都是不带 `maxsize` 的 `queue.Queue`（`vllm/v1/engine/core.py:1362-1363`）；
  client 的 `outputs_queue` 是不带 `maxsize` 的 `asyncio.Queue`（`vllm/v1/engine/core_client.py:1209`，同步 client 在 `:1008` 是 `queue.Queue`）。
- `make_zmq_socket` 给 `PULL`、`DEALER`、`ROUTER` 把 `RCVHWM` 设为 0，给 `PUSH`、`DEALER`、`ROUTER` 把 `SNDHWM` 设为 0（`vllm/utils/network_utils.py:371-377`），
  0 表示 ZMQ 不设水位线，所以套接字这一层也不会因为队列长而丢弃或阻塞。
- `RequestOutputCollector` 只有一个 `output` 槽：生产者领先时，`put` 把新输出并入已有的那一个（`RequestOutput.add`，`aggregate` 只在 delta 模式为真，
  否则用新输出替换同序号的旧输出），消费者 `get` 取走并清空（`vllm/v1/engine/output_processor.py:54-100`，`vllm/outputs.py:211-255`）。
- Engine→Worker 的广播走 `MessageQueue`；写端 `acquire_write` 发现下一个槽还没被所有 reader 读完就循环等待（`vllm/distributed/device_communicators/shm_broadcast.py:732-760`），
  槽数默认是 10（`MessageQueue.__init__` 的 `max_chunks: int = 10`，`:507`）。这是这张图里唯一会让写端真正等待的位置（Worker 回 EngineCore 的 response 队列也是同一种 ring buffer，见第 03 章）。

所以“满了怎么办”在不同位置的答案不一样：`check_admission` 满了是**拒绝**，没有上限的队列是**积压**（内存增长、延迟变长，但不报错），
Collector 是**合并**，ring buffer 是**让写端等**。排查延迟时，先问请求卡在哪一类位置，再去看对应的指标。

API 的 admission control 通过 `max_num_queued_reqs` 和
`max_num_queued_tokens` 在进入 EngineCore 前拒绝过量请求，但两项指标的范围不同。
当启动地址包含 `mp_admission_counters` 时，`AsyncLLM` 创建 `SharedAdmissionStats`，
请求数检查读取共享计数；否则回退到本地 OutputProcessor 未完成请求数。
这是近似全局上限，不是跨进程原子预留：其他 API 进程可能同时通过检查。

queued-token 检查仍调用本地 `get_num_queued_tokens()`，不能因为请求数共享了就推断
token 数也跨 API 进程共享。它在现有计数已达到阈值时拒绝，并非先把新 prompt 长度
加入计数后进行精确预算。`get_num_queued_tokens()` 当前
对仍在 prefill 的请求使用完整 `prompt_len`，不会减去 chunked prefill 已完成部分或
prefix hit，因为这些进度要到 prefill 完成后才传播给 API Server。这是保守估计：可能
更早拒绝，但有利于保护 TTFT。

不要把 admission queue、Scheduler waiting queue 和 output queue 混成一个指标。它们的
容量、所有者和拥塞表现完全不同。

[源码] `vllm/v1/engine/async_llm.py` - `AsyncLLM.check_admission`

[源码] `vllm/v1/engine/output_processor.py` - `get_num_queued_tokens`

[源码] `vllm/v1/engine/admission_control.py` - `SharedAdmissionStats`

## 2.16 配置怎样贯穿全系统

入口层接收大量 Python 参数或 CLI flags，最终汇聚到 `VllmConfig`。各层保存自己关心的
子配置引用：

```mermaid
flowchart LR
    A[CLI / LLM kwargs] --> B[EngineArgs / AsyncEngineArgs]
    B --> C[create_engine_config]
    C --> V[VllmConfig]
    V --> M[ModelConfig]
    V --> S[SchedulerConfig]
    V --> K[CacheConfig]
    V --> P[ParallelConfig]
    V --> CP[CompilationConfig]
    V --> O[ObservabilityConfig]
```

> **读图方法：** 阅读“配置怎样贯穿全系统”这张流程图时，先把方框看成对象或状态，把箭头看成数据或控制的移动。第一遍从左向右建立顺序，第二遍再核对分支发生的条件。

这种设计让 Model Runner 可以直接读取 compilation config，Scheduler 可以读取 cache
和 parallel 约束，而不必在每层 constructor 上继续增加参数。但代价是：一个配置字段
可能在很远的模块才真正产生行为，读源码必须同时追踪“字段定义、默认值推导、校验、
消费位置”。第 03 章会完整展开配置初始化顺序。

## 2.17 最小可运行生命周期模拟器

本项目提供一个不依赖 PyTorch、CUDA 或 vLLM 安装的教学模拟器：

- [源码](../../examples/ch02_request_lifecycle.py)
- [测试](../../tests/test_ch02_request_lifecycle.py)

它保留了以下真实边界：

```text
raw prompt
  -> EngineInput
  -> EngineCoreRequest
  -> CoreRequestState
  -> SchedulerOutput
  -> ModelRunnerOutput
  -> EngineCoreOutput
  -> RequestOutput
```

它故意省略真实模型、KV Cache、batch packing 和 ZMQ，只用确定性的 token plan 展示：

- 离线请求怎样持续 step 并只返回最终结果。
- 在线请求怎样每 step 返回 delta。
- internal ID 怎样用于 core，external ID 怎样返回用户。
- abort 怎样产生最终完成状态。
- 前端为什么应在提交到 core 前注册本地 state。

运行：

```bash
python3 examples/ch02_request_lifecycle.py
python3 -m unittest tests/test_ch02_request_lifecycle.py
```

2026-09-07 在本项目环境中的输出：

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

八个测试覆盖以下边界：

1. 离线结果保持提交顺序，外部 ID 的字典顺序不决定返回顺序。
2. 在线结果按 token delta 交付。
3. core 使用内部 ID，用户仍看到外部 ID。
4. 前端 state 在 core submission 前已经存在。
5. prompt 必须为生成留出至少一个 token；请求输出上限不预留窗口，生成在输出上限或上下文上限先到达时停止。
6. 重复外部 ID 不会混淆两次提交的输出顺序。
7. `max_tokens` 大于六个 token 的初始响应计划时，不会被意外提前截断。
8. 最后一个请求取消后，仍交付 abort 输出并清理前端状态。

教学响应计划会循环使用固定六个 token，直到达到请求上限；它只验证长度和状态推进，
没有语言生成质量含义。提交顺序由内部 ID 对应的序号记录，而不是对用户 ID 做排序。

[源码] `vllm/entrypoints/common/offline.py` - `OfflineInferenceMixin._add_request`、`_run_engine`

真实 `LLM.generate` 用递增数值 ID 记录输入顺序，最终按该数值排序；本项目教学程序
允许自定义外部 ID，因此需要另记提交序号。两种实现的目标都是保持输入顺序。

v0.32.0 不再把完整 `max_tokens` 从上下文窗口中预留。必须分清三个边界：入口校验 prompt 本身；在线路径用 `get_max_tokens` 决定输出上限；执行侧 `check_stop` 按长度结束。随书模型把后两处收成同一个教学上限：

$$
\min\bigl(\mathtt{max\_tokens},\,\mathtt{max\_model\_len}-\texttt{prompt\_len}\bigr)
$$

**白话：**一次生成最多能写出多少个新 token，取决于“请求自己要多少”和“窗口里还剩多少”里更小的那个。窗口已经装不下完整请求上限时，不会在入口按 $\texttt{prompt\_len}+\mathtt{max\_tokens}$ 直接拒绝，而是把输出上限削到剩余窗口。

**其中：**$\mathtt{max\_tokens}$ 是请求（或默认）给出的输出数量上限（个 token）；$\mathtt{max\_model\_len}$ 是模型上下文窗口（个 token）；$\texttt{prompt\_len}$ 是已经占用窗口的 prompt 长度（个 token）。三者都是非负整数（个 token）。

**怎么算出来：**执行侧 `check_stop` 分别检查总 token 数是否达到 `max_model_len`、输出 token 数是否达到请求上限，任一成立就结束（`vllm/v1/core/sched/utils.py:114-119`）。因此还能写出的个数不会超过剩余窗口 $\mathtt{max\_model\_len}-\texttt{prompt\_len}$，也不会超过 $\mathtt{max\_tokens}$，教学上取二者最小。
在线 `get_max_tokens` 在此之外还会与覆盖值 `override_max_tokens`、平台限制 `platform_max_tokens` 一起取最小；若设了 `truncate_prompt_tokens`，参与相减的输入长度还会先被截到该上限（`vllm/entrypoints/serve/utils/api_utils.py:234-270`）。
离线 `InputProcessor` 只在请求未指定 `max_tokens` 时，把它补成 `max_model_len - prompt_len`（`vllm/v1/engine/input_processor.py:471-475`）。
算例（教学数字）：窗口为 5、prompt 为 4、请求 $\mathtt{max\_tokens}=100$，则 $\min(100,\,5-4)=1$，并不因“$4+100>5$”拒绝，而最多生成 1 个 token。

**单位检查：**三项都是 token 个数，相减得到剩余个数，再与请求上限取最小，单位一致。

**核对：**入口校验只拒绝“prompt 本身超过窗口”，以及生成路径上 prompt 长度恰好等于 `max_model_len` 的情况（至少还需 1 个输出 token 的位置都没有，`vllm/v1/engine/input_processor.py:563-597`，生成任务在 `:588-597`）。
`check_stop` 的长度条件是 `request.num_tokens >= max_model_len or request.num_output_tokens >= request.max_tokens`（`vllm/v1/core/sched/utils.py:114-119`）。
随书模型保留原请求上限，在 Core 的确定性响应计划中取这两种停止边界的最小值；这是等价的教学简化，
不模拟生产 tokenization、EOS 或异步停止的全部路径。Core 也保留教学用终态记录，不模拟完整后台资源回收。

[源码] `vllm/entrypoints/serve/utils/api_utils.py` - `get_max_tokens`

[源码] `vllm/v1/engine/input_processor.py` - `InputProcessor._validate_prompt_len`、`process_inputs`

[源码] `vllm/v1/core/sched/utils.py` - `check_stop`

这个模拟器不能证明 vLLM 性能，也不能代替 GPU 集成测试。它的作用是让读者在进入
复杂源码前，先看见对象所有权和状态转换。

## 2.18 源码逐跳核对

下面把在线 happy path 的每条边落实到当前 revision。

| 跳转 | 源码证据 | 关键观察 |
|---|---|---|
| HTTP request -> render | `OpenAIServingChat._create_chat_completion` | 调用 `render_chat_request`，生成 `engine_inputs` |
| render -> EngineInput | `OnlineRenderer.render_chat`、`BaseRenderer.render_chat_async` | template、tokenize、process for engine |
| EngineInput -> engine generator | `OpenAIServingChat._create_chat_completion` | 调用 `engine_client.generate(...)` |
| generator -> EngineCoreRequest | `AsyncLLM.add_request`、`InputProcessor.process_inputs` | 校验并构造 request |
| 注册前端 state | `AsyncLLM._add_request` | 第一次 await 前调用 `OutputProcessor.add_request` |
| request 跨进程 | `AsyncMPClient.add_request_async` | msgpack + ZMQ multipart |
| wire request -> Request | `EngineCoreProc.process_input_sockets`、`preprocess_add_request` | input thread 中完成反序列化与初始化 |
| Request -> Scheduler | `EngineCore.add_request` | `scheduler.add_request(request)` |
| Scheduler -> Worker | `EngineCore.step` | `schedule()` 后 `execute_model(non_block=True)` |
| Worker -> Scheduler | `EngineCore.step` | `update_from_output(scheduler_output, model_output)` |
| core output 跨进程 | `EngineCoreProc.process_output_sockets` | msgpack/ZMQ，按 `client_index` 发送 |
| output -> text | `OutputProcessor.process_outputs` | detokenize、stop、logprobs、完成清理 |
| text -> per-request stream | `AsyncLLM._run_output_handler` | collector `put()` |
| stream -> SSE | `chat_completion_stream_generator` | `data: JSON\n\n` |

## 2.19 设计取舍

> **本节先看：** 下面不只说明代码怎样写，还解释为什么采用这种边界。比较每个方案时，同时考虑正确性、复杂度、显存和性能。

### 为什么 tokenization 留在前端

> **本节先看：** 下面先给出“为什么 tokenization 留在前端”的结论清单。先理解每一项为什么存在，再记参数名或实现细节。

- API Server 已经拥有 tokenizer 和用户协议上下文。
- 阻塞预处理可以放进前端线程池，不占用 EngineCore busy loop。
- EngineCore 只接收更稳定的 token/request 契约。
- 多 API Server 可以扩展 CPU 预处理能力。

代价是 tokenizer、Renderer cache 和 OutputProcessor state 占用前端 CPU/内存，而且
多前端时需要 request 路由和 `client_index`。

### 为什么 Scheduler 与模型执行分层

Scheduler 操作 request 状态、预算和 block metadata；Model Runner 操作 tensor 和 device
执行。分层使调度算法可以在 CPU 上快速变化，也让不同 accelerator/backend 复用同一
请求语义。

### 为什么输出处理回到前端

> **本节先看：** 下面先给出“为什么输出处理回到前端”的结论清单。先理解每一项为什么存在，再记参数名或实现细节。

- detokenization 和 stop string 是字符串/协议问题。
- 在线输出需要每请求异步队列和 HTTP streaming。
- 避免 EngineCore 被 JSON、tool parser 等复杂前端逻辑拖慢。

代价是 EngineCore 和前端都拥有一份与请求相关的状态，abort 和完成必须双边一致清理。

### 为什么新请求与 cached request 在 `SchedulerOutput` 中分开

Worker 会缓存请求的稳定数据。第一次调度发送完整 `NewRequestData`，后续 step 只发送
token、block 和计数差异，降低 CPU 序列化及进程通信成本。第 07 章会展开 persistent
batch 如何消费这些增量。

## 2.20 常见误解

> **本节先看：** 下面集中修正最容易由名称或旧版本经验造成的误解。先判断自己原来的理解，再用后面的源码事实校正。

### 误解 1：`LLMEngine` 就是 Scheduler

不是。`LLMEngine` 是前端兼容门面，持有 InputProcessor、OutputProcessor 和
EngineCoreClient。Scheduler 位于 EngineCore 内部。

### 误解 2：API Server 只是 HTTP 转发器

不是。它承担模型检查、chat template、tokenization、多模态预处理、参数转换、输出
协议、tool/reasoning parsing、streaming 和取消传播。

### 误解 3：一次请求只调用一次模型

通常不是。请求至少经历 prefill，随后每个 decode step 再次参与调度和执行，直到结束。

### 误解 4：所有在线对象都在一个 asyncio event loop

不是。API Server 有 asyncio task；EngineCoreProc 有独立进程、input/output 线程和
busy loop；Worker 是否独立进程取决于 executor backend；GPU kernel 在 device 上执行。

### 误解 5：每张 GPU 必然对应一个额外 Worker 进程

当前单 rank `UniProcExecutor` 直接在 EngineCore 进程内创建 Worker。应区分逻辑 Worker
和进程部署。

### 误解 6：EngineCoreOutput 已经是最终文本

不是。它的核心仍是 `new_token_ids`。OutputProcessor 才做增量 detokenization 和文本
stop 检查。

### 误解 7：abort 只需要停止 HTTP stream

不够。还必须释放 OutputProcessor state、Scheduler request、KV block 及相关 connector
资源。

### 误解 8：官方设计文档永远等同于当前代码

设计文档提供总体意图，但可能保留历史类名或概括部署。当前行为要以锁定 revision 的
构造和调用路径为准。本章对 UniProc Worker 进程边界的修正就是一个例子。

## 2.21 阅读与调试指南

遇到一次请求“卡住”“没有输出”或“取消后仍占显存”，不要从 CUDA kernel 开始。按
数据形态逐层定位：

| 观察点 | 要确认什么 | 典型源码 |
|---|---|---|
| API 入口 | schema、model、max_tokens 是否通过 | OpenAI serving |
| Renderer | chat template 后 prompt/token 是否正确 | `renderers/` |
| InputProcessor | 长度、token、params、internal ID | `vllm/v1/engine/input_processor.py` |
| EngineCoreClient | ADD 是否发送、client 是否 alive | `vllm/v1/engine/core_client.py` |
| Engine input thread | 是否 decode/preprocess 并入队 | `vllm/v1/engine/core.py::process_input_sockets` |
| Scheduler | request 在 waiting/running/finished 哪一处 | 第 04 章 |
| Executor | `SchedulerOutput` 是否被执行 | 第 07 章 |
| Engine output thread | output 是否按 client index 发回 | `process_output_sockets` |
| OutputProcessor | state 是否存在、是否被 stop/abort | `vllm/v1/engine/output_processor.py` |
| SSE formatter | 是否产生 JSON chunk 与 `[DONE]` | chat serving |

一个实用规则是始终打印或记录同一个 internal request ID。外部 ID 可能映射到多个 child，
只用外部 ID 很难区分并行 sampling 的状态。

### v0.32.0：一轮多个 token 的停止边界

一次 EngineCore 输出可能含多个 token，不能假定输出处理仅删除文本尾部。
`OutputProcessor.process_outputs` 在 detokenizer 命中 stop string 后调用 `_trim_surplus_tokens`，
同步裁剪多余 token IDs、logprobs、sampling mask 与 routed-expert 片段，再构造用户输出。
这保持不同输出字段的长度边界一致；教学模型每步一个 token，不覆盖该多 token 回退场景。

[源码] `vllm/v1/engine/output_processor.py` - `process_outputs`、`_trim_surplus_tokens`

[测试] `tests/v1/engine/test_output_processor.py`

## 2.22 静态验证与运行限制

本章完成了三类验证：

1. **源码静态核对**：从两个入口逐跳走到 EngineCore、执行层和输出层。
2. **上游测试交叉验证**：检查 collector merge、abort、final-step abort 竞争等测试。
3. **CPU 教学实验**：运行生命周期模拟器及 8 个单元测试。

当前机器没有安装可用的 PyTorch/CUDA/vLLM GPU runtime，因此未完成以下动态验证：

- 启动真实 OpenAI-compatible server 并采集 PID/thread/ZMQ 拓扑。
- 运行真实模型，记录一个请求跨进程的时间线。
- 在 GPU forward 中途取消请求，观察显存和 KV usage 回落。
- 对比 `InprocClient`、`SyncMPClient` 和 `AsyncMPClient` 的真实调用栈。

因此本章 `content_complete=true`，但 `runtime_verified=false`，状态保持 `draft`。

## 2.23 本章小结

一个生成请求不是“调用模型然后返回字符串”，而是一条跨层生命周期：

```text
用户协议
  -> Renderer
  -> EngineInput
  -> InputProcessor
  -> EngineCoreRequest
  -> EngineCoreClient / IPC
  -> Request / Scheduler
  -> SchedulerOutput
  -> Executor / Worker / GPU
  -> ModelRunnerOutput
  -> EngineCoreOutput
  -> OutputProcessor
  -> RequestOutput
  -> Python return 或 SSE
```

离线与在线路径共享 EngineCore 之后的大部分后端。默认两者的 EngineCoreProc 都在
后台推进；离线前端循环调用 `LLMEngine.step()` 同步取结果，在线前端用异步 output
handler 分发结果。只有显式关闭多进程的离线 `InprocClient` 路径，调用线程才直接
执行 `EngineCore.step_fn()`。协议面上另外还有 `/v1/decisions` 与 `/v1/systemone`：
它们把预先给定的选项打成结构化判别，仍经 EngineClient 进入同一套后端，细节见上文 Decisions 短节。

[源码] `vllm/v1/engine/llm_engine.py` - `LLMEngine.from_engine_args`、`step`

[源码] `vllm/v1/engine/core_client.py` - `InprocClient.get_output`、`SyncMPClient.get_output`

理解 vLLM 源码时必须同时问四个问题：当前对象拥有什么状态、运行在哪个并发边界、
输入输出是什么数据形态、完成或取消时由谁清理。只记住类名不能回答这些问题。

## 2.24 自检问题

第一遍先完成前 5 题；余下题目是第二遍源码阅读的扩展题库，不要求一次做完。

> **本节先看：** 建议先不看答案，用自己的话回答下面的问题。能够解释原因、画出数据流并指出源码位置，才算真正理解。

1. Renderer 与 InputProcessor 的职责差异是什么？
2. 为什么 `EngineCoreRequest` 到 EngineCore 后还要转成 `Request`？
3. `LLM.generate()` 为什么能保持输入顺序，即使请求完成顺序不同？
4. 在线路径为什么先注册 OutputProcessor state，再 await 跨进程提交？
5. `InprocClient` 与 `AsyncMPClient` 的核心差异是什么？
6. 为什么不能断言“一张 GPU 一定多一个 Worker 进程”？
7. `SchedulerOutput` 与 `EngineCoreOutput` 分别描述哪个方向的数据？
8. OutputProcessor 在什么情况下会反向要求 EngineCore abort？
9. 客户端取消时为什么需要前后端双边清理？
10. `RequestOutputCollector` 为什么会合并 DELTA 输出？

## 2.25 源码追踪题

> **本节先看：** 这些题目要求把概念重新落到代码。先从给出的入口搜索定义和调用点，再记录对象、状态与边界，不要只找到同名符号。

1. 从 `LLM.generate()` 出发，找到真正的 `while has_unfinished_requests()` 循环。
2. 从 `OpenAIServingChat._create_chat_completion()` 出发，找到
   `AsyncLLM.generate()` 的调用点。
3. 追踪一个 `EngineCoreRequest` 如何被 msgpack 编码、在 input thread 解码并转成
   `Request`。
4. 找到 `EngineCore.step()` 中处理 abort 与 `update_from_output()` 的先后顺序，并说明
   对应测试。
5. 找到 `OutputProcessor.process_outputs()` 中在线 queue 与离线 return list 的分叉。
6. 从 `ParallelConfig` 找到单 rank默认 `"uni"`，再追踪到
   `UniProcExecutor._init_executor()`。

## 2.26 参考答案

<details>
<summary>1. Renderer 与 InputProcessor</summary>

Renderer 面向用户语义，负责模板化、tokenization 和多模态 processing，产出
`EngineInput`。InputProcessor 面向 EngineCore 契约，负责参数与长度校验、request-local
参数补全以及构造 `EngineCoreRequest`。
</details>

<details>
<summary>2. 为什么还要构造 Request</summary>

`EngineCoreRequest` 是可序列化的提交消息；`Request` 是 EngineCore 内持续变化的运行
状态，包含 status、computed/output token、block hash、preemption、in-flight token 等。
</details>

<details>
<summary>3. 离线返回顺序</summary>

`OfflineInferenceMixin._run_engine()` 只收集 finished output，循环结束后按整数 request ID
排序。request ID 按输入遍历顺序由 counter 生成。
</details>

<details>
<summary>4. 为什么先注册前端状态</summary>

这样并发 task、取消逻辑和可能很快返回的 output 都能找到本地 RequestState；源码明确
要求在第一次 await 前完成注册。
</details>

<details>
<summary>5. Client 差异</summary>

`InprocClient` 直接调用同进程 EngineCore，并由调用者触发 step；`AsyncMPClient` 使用
msgpack/ZMQ 与后台 EngineCoreProc 通信，通过 asyncio queue 异步接收输出。
</details>

<details>
<summary>6. Worker 进程边界</summary>

单 rank 默认 `UniProcExecutor`，它在 EngineCore 进程内实例化 WorkerWrapperBase。只有
mp、Ray 等 backend 才把 Worker 放进其他进程。
</details>

<details>
<summary>7. 两个 Output</summary>

`SchedulerOutput` 从控制面发往执行侧，描述本 step 要算哪些 request/token/block；
`EngineCoreOutput` 从 EngineCore 发回前端，描述每个请求新产生的 token 和完成状态。
</details>

<details>
<summary>8. 前端触发 abort</summary>

当增量 detokenization 发现 stop string，而 EngineCore 尚未把请求标记 finished 时，
OutputProcessor 生成用户完成输出，并把内部 ID 放入 `reqs_to_abort`。
</details>

<details>
<summary>9. 双边清理</summary>

前端拥有 collector、detokenizer 和 ID mapping；后端拥有 Scheduler、KV Cache 和 Worker
状态。只清理一侧都会产生泄漏或无效计算。
</details>

<details>
<summary>10. DELTA 合并</summary>

当 producer 快于 consumer 时，合并多个 delta 能限制前端输出积压和 task 切换，同时
保留已经生成的文本、token、logprobs 以及不同 choice index。
</details>

## 2.27 源码与资料索引

> **本节先看：** 下面按主题整理本章使用的证据入口。需要复核结论时，优先按路径和符号定位，不依赖可能漂移的行号。

- `vllm/entrypoints/llm.py`
  - `LLM.generate`
- `vllm/entrypoints/common/offline.py`
  - `OfflineInferenceMixin._add_completion_requests`
  - `OfflineInferenceMixin._run_completion`
  - `OfflineInferenceMixin._run_engine`
- `vllm/entrypoints/openai/chat_completion/serving.py`
  - `OpenAIServingChat.create_chat_completion`
  - `OpenAIServingChat._create_chat_completion`
  - `OpenAIServingChat.chat_completion_stream_generator`
  - `OpenAIServingChat.chat_completion_stream_generator_raw`
  - `OpenAIServingChat.chat_completion_full_generator`
- `vllm/entrypoints/openai/decisions/api_router.py`、`serving.py`、`protocol.py`
  - `create_decision`、`OpenAIServingDecisions.create_decisions`
- `vllm/entrypoints/generate/structured_decisions/api_router.py`、`serving.py`
  - `/v1/systemone`、`ServingStructuredDecisions`
- `docs/models/structured_decisions_models.md`
- `vllm/renderers/base.py`
  - `BaseRenderer.render_cmpl`
  - `BaseRenderer.render_chat`
  - `BaseRenderer.validate_token_ids`
  - `BaseRenderer.queue_mm_cache_invalidations`
- `vllm/renderers/online_renderer.py`
  - `OnlineRenderer.render_chat`
- `vllm/v1/engine/input_processor.py`
  - `InputProcessor.process_inputs`
  - `InputProcessor.assign_request_id`
- `vllm/v1/engine/llm_engine.py`
  - `LLMEngine.add_request`
  - `LLMEngine.step`
- `vllm/v1/engine/async_llm.py`
  - `AsyncLLM.add_request`
  - `AsyncLLM.generate`
  - `AsyncLLM._run_output_handler`
  - `AsyncLLM.abort`
- `vllm/v1/engine/core_client.py`
  - `EngineCoreClient.make_client`
  - `InprocClient`
  - `SyncMPClient`
  - `AsyncMPClient`
- `vllm/v1/engine/core.py`
  - `EngineCore.preprocess_add_request`
  - `EngineCore.add_request`
  - `EngineCore.step`
  - `EngineCoreProc.run_busy_loop`
  - `EngineCoreProc.process_input_sockets`
  - `EngineCoreProc.process_output_sockets`
  - `EngineCoreProc.pause_generation`
  - `EngineCoreProc._reject_add_while_paused`
- `vllm/v1/engine/output_processor.py`
  - `RequestOutputCollector`
  - `StreamingInputOutputCollector`
  - `OutputProcessor.process_outputs`
  - `OutputProcessor.abort_requests`
- `vllm/v1/request.py`
  - `Request`
  - `RequestStatus`
- `vllm/v1/core/sched/output.py`
  - `SchedulerOutput`
- `vllm/v1/executor/abstract.py`
  - `Executor.get_class`
- `vllm/v1/executor/uniproc_executor.py`
  - `UniProcExecutor`
- `tests/v1/engine/test_output_processor.py`
- `tests/v1/engine/test_async_llm.py`
- `tests/v1/engine/test_abort_final_step.py`
- `docs/design/arch_overview.md`
- [本项目第 2 章模拟器](../../examples/ch02_request_lifecycle.py)
- [本项目第 2 章测试](../../tests/test_ch02_request_lifecycle.py)
- [第 2 章实验记录](../../experiments/ch02-request-lifecycle.md)

## 完成状态

> **本节先看：** 这里区分正文完成、静态源码核对和真实 GPU 运行验证。没有执行过的实验不会因为正文完整就被标记为已验证。

- [x] 对比 offline 与 online 两条入口。
- [x] 逐跳追踪 Renderer、InputProcessor、EngineCoreClient、EngineCore 和 OutputProcessor。
- [x] 区分七种关键数据形态。
- [x] 标出 asyncio、线程、ZMQ、进程和 GPU 边界。
- [x] 修正单 GPU `UniProcExecutor` 下 Worker 并非独立进程的过度概括。
- [x] 覆盖 stop string、client cancel、final-step abort 和 output handler error。
- [x] 提供可运行生命周期模拟器和 8 个单元测试。
- [ ] 在真实 NVIDIA GPU 上采集 online/offline 运行 trace。

本章内容已经完整，`content_complete=true`。由于当前环境没有真实 vLLM GPU runtime，
`runtime_verified=false`，待 GPU 动态验证，状态为 `draft`。

