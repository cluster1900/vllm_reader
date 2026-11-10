# 03-engine-core：v0.33.0 升级复核（B4）

- 结论：**已更新（有实质影响）**。
- Source：`4ba787ec9959e566d3b587a796dff07673324da5`（v0.33.0）；旧正文基线 `5df63a7203cdb9328d824a27b682e77c7b9928be`（v0.32.0）；Reader base：`ed2676df`。
- 日期：2026-11-10；静态源码验证，章节保持 draft / runtime_verified=false。
- 跟踪：[bot-hub #127](https://github.com/cluster1900/bot-hub/issues/127)。
- 范围：只写 v0.33.0 祖先链里真实存在的变化（每个提交均用 `git merge-base --is-ancestor <commit> 4ba787ec9959` 核对）。原“main 上的变化（未发版）”注记里的 5 个提交均为祖先，已并入正文并删除注记，没有遗留的未发版注记。

## 证据与修订

| 固定源码 | 符号/入口 | 上游提交 | 结论与正文修改 |
|---|---|---|---|
| [vllm/v1/engine/core.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/core.py) | `EngineCore._initialize_kv_caches` | 3e75ccba2 (#62330)、30aded62d (#52921) | 全局分组 + DCP interleave 前移到 profile 之前；CPU offload 块数回填。3.6.2 步骤改写（9 步）、流程图加节点，原 #62330 注记并入正文。 |
| [vllm/v1/engine/core.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/core.py) | `EngineCoreProc.pause_generation`、`_reject_add_while_paused`、`resume_scheduler` | d5b36699b (#51488) | 用户暂停的 abort/wait 拒绝新请求。3.12 模式表分两种入口、状态图加 `PAUSED_NEW_REJECT`、新增时序图，原注记并入正文。 |
| [vllm/v1/engine/core.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/core.py) | `EngineCore._set_kv_connector_handshake_metadata`、`wake_up` | afe2183f2 (#59635) | 唤醒后重新下发 KV connector 握手元数据；3.12 sleep 段补充。 |
| [vllm/v1/engine/core.py、vllm/utils/watch_dog.py、vllm/config/watchdog_config.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/core.py、vllm/utils/watch_dog.py、vllm/config/watchdog_config.py) | `ThreadedWatchDog`、`start_watch_dog`、`WatchdogConfig` | 1c8799197 (#55700) | 新增 3.10.4（默认关闭；feed/disable/handle_failure；超时判定公式与算例；流程图）；3.13 补 SIGTERM 转储。 |
| [vllm/v1/engine/core.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/core.py) | `DPEngineCoreProc.add_request` | a94c31571 (#59559) | wave 由引擎唤醒，前端不再发 FIRST_REQ。新增 3.4.4（时序图）。 |
| [vllm/v1/engine/coordinator.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/coordinator.py) | `DPCoordinatorProc` | a94c31571 (#59559) | 协调器不再处理前端 wave 消息，收到则记 error 日志。 |
| [vllm/v1/engine/core_client.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/core_client.py) | `DPAsyncMPClient`、`DPLBAsyncMPClient`、`MPClient._apply_ready_response` | a94c31571 (#59559)、dd9e07a25 (#43202)、146350155 (#58942) | `first_req_*` 改名 `scale_*`；外部 LB 的 Elastic EP 协调；streaming 会话 abort 广播；READY 新字段。3.4.4、3.2.2 补充。 |
| [vllm/v1/engine/async_llm.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/async_llm.py) | `AsyncLLM._run_output_handler`、`_add_request`、`pause_generation` | 70806b631 (#62164)、d5b36699b (#51488)、dd9e07a25 (#43202) | 输出 handler 先排队 mm 缓存失效；`GenerationError` 视为请求级失败并回收 n>1 同胞；退役 rank 的 health 处理。3.3.2、3.4.4、3.12。 |
| [vllm/v1/engine/core.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/core.py) | `EngineCoreProc.process_input_sockets` | 88077b746 (#49625) | `VLLMValidationError` → `validation_error`。3.5.1 补充。 |
| [vllm/v1/engine/core_client.py、core.py、llm_engine.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/core_client.py、core.py、llm_engine.py) | `profile(profiler_kwargs=…)` | dd5e75449 (#51974) | profile payload 透传。3.7 补充。 |
| [vllm/distributed/device_communicators/shm_broadcast.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/device_communicators/shm_broadcast.py) | `SpinCondition.record_read` | 83c078d95 (#63165) | 忙 reader 定期 poll notify socket。3.5 补充。 |
| [vllm/v1/worker/gpu/spec_decode/speculator.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/spec_decode/speculator.py) | draft metadata `max_seq_len` | 636da2476 (#60513) | auto-fit 后同步 draft 长度。3.2.2，原注记并入正文。 |

## 不修改正文的 v0.33.0 变化

- `vllm/config/parallel.py` 把 `DPEngineCoreProc` 对 DP 状态同步的调用由 `ParallelConfig.sync_dp_state(...)` 改为 `parallel_config.sync_dp_state(...)`（同批 CPU all-reduce 相关改动 #61767、#63016）：属于第 09 章。
- `DPEngineCoreProc` 输出路径改用 `zmq.Context.instance()`（避免每次新建 context）：实现细节，不影响本章叙述。
- `ParallelConfig.enable_shm_tensor_arena` 默认值仍为 `False`，3.5 的结论不变（已复核）。
- Elastic EP 外部 LB 的完整流程：留给第二波的新主题章节，本章只在 3.4.4 说明接口边界。

## 对其他批次的提示

- `AsyncScheduler._update_request_with_output` 在 v0.33.0 被删除并上移到 `Scheduler` 基类（#58668）；第 08 章对该覆盖方法的描述需要同步（B2）。
