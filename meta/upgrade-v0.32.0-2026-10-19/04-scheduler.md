# 04-scheduler：v0.32.0 升级复核

- 结论：**已更新（有实质影响）**。
- Source: `5df63a7203cdb9328d824a27b682e77c7b9928be`；Reader base: `5f78d105690c1968885abb8c0264cd98fb7b9a18`。
- 日期：2026-10-19；静态源码验证，章节保持 draft / runtime_verified=false。

## 证据与修订

| 固定源码 | 符号/入口 | 结论与正文修改 |
|---|---|---|
| [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py) | `schedule / get_num_unfinished_requests` | PAUSED_NEW 优先 drain kv_holding_waiting；普通 waiting 只在 UNPAUSED 准入，流式等待输入不阻塞 drain。 |
| [tests/v1/core/test_scheduler.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/tests/v1/core/test_scheduler.py) | `test_pause_new_drains_async_kv_loads / test_pause_new_ignores_streaming_session_waiting_for_input` | 测试证据支持 KV-holding waiting 与普通 waiting 分开，不把旧/new 标签当资源所有权。 |
| [vllm/v1/core/sched/output.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/output.py) | `SchedulerOutput` | num_common_prefix_blocks 仅保留空默认兼容字段；block_table_updates 删除。 |
| [vllm/v1/core/kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_manager.py) | `get_num_common_prefix_blocks` | 返回全零的树外兼容接口，不代表 cascade attention 仍执行。 |

## 教学示例与验证边界

ch04 教学程序未模拟 KV-holding waiting、pause 或 cascade，既有 token-budget/抢占模型无影响；本次重跑其测试，不把它当生产暂停路径验证。

统一实际测试结果及逐文件变更覆盖见[总报告](README.md)。旧实验报告不作本次运行证明。
