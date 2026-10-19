# 03-engine-core：v0.32.0 升级复核

- 结论：**已更新（初始化退出、utility 汇总与 IPC 边界）**。
- Source: `5df63a7203cdb9328d824a27b682e77c7b9928be`；Reader base: `5f78d105690c1968885abb8c0264cd98fb7b9a18`。
- 日期：2026-10-19；静态源码验证，章节仍为 draft / runtime_verified=false。

## 证据与修订

| 固定源码 | 符号/入口 | 结论与正文修改 |
|---|---|---|
| [vllm/v1/engine/core.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/core.py) | `EngineCoreProc.run_engine_core` | 初始化窗口保存 shutdown_requested，构造后才安装 SignalCallback；不直接从 signal handler 锁队列。 |
| [vllm/v1/engine/core_client.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/core_client.py) | `MPClient.__init__ / call_utility_all_async / call_utility_consensus_async` | bind address 与实际 endpoint 分开；prefix reset/wake 聚合 all，状态查询要求一致。 |
| [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py) | `Scheduler.schedule / get_num_unfinished_requests` | PAUSED_NEW 能 drain 持有 KV 的 waiting；普通 waiting 仍暂停，不能仅按旧/新或 waiting/running 名称解释。 |
| [vllm/config/parallel.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/parallel.py) | `ParallelConfig.enable_shm_tensor_arena` | 默认 False，experimental/opt-in；release 摘要的默认开启说法与锁定源码冲突。 |
| [vllm/v1/executor/multiproc_executor.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/executor/multiproc_executor.py) | `MultiprocExecutor._init_executor` | 将 enable_shm_tensor_arena 传给 Engine→Worker 广播队列，不是替代前端全部 IPC。 |
| [vllm/distributed/device_communicators/shm_broadcast.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/distributed/device_communicators/shm_broadcast.py) | `MessageQueue` | arena 要求全部 reader node-local；不把远端消息路径写成共享内存。 |
| [vllm/entrypoints/common/offline.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/entrypoints/common/offline.py) | `OfflineInferenceMixin._run_engine` | 第03章同步离线示例调用链使用迁移后的模块。 |

## 教学示例与验证边界

ch03 无 prompt/context 维度，max_tokens 仅为输出工作量上限，不是上下文预留。新增 batch queue 回归验证不会多提交/多结算 token。该教学模型仍不区分生产 KV-holding waiting，已保留简化说明。

统一测试结果和逐文件差异覆盖见[总报告](README.md)。历史实验记录不作为此次运行证据。
