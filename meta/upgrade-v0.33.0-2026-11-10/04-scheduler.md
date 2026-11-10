# 04-scheduler：v0.33.0 升级复核（B4）

- 结论：**已更新（有实质影响）**。
- Source：`4ba787ec9959e566d3b587a796dff07673324da5`（v0.33.0）；旧正文基线 `5df63a7203cdb9328d824a27b682e77c7b9928be`（v0.32.0）；Reader base：`ed2676df`。
- 日期：2026-11-10；静态源码验证，章节保持 draft / runtime_verified=false。
- 跟踪：[bot-hub #127](https://github.com/cluster1900/bot-hub/issues/127)。
- 范围：只写 v0.33.0 祖先链里真实存在的变化（每个提交均用 `git merge-base --is-ancestor <commit> 4ba787ec9959` 核对）。原“main 上的变化（未发版）”注记里的 5 个提交均为祖先，已并入正文并删除注记，没有遗留的未发版注记。

## 证据与修订

| 固定源码 | 符号/入口 | 上游提交 | 结论与正文修改 |
|---|---|---|---|
| [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py) | `Scheduler.schedule`（running 循环） | ceb01b6dd (#61976) | PRIORITY 自我抢占改 `continue`。4.8 流程图加节点，4.8.2 注记并入正文。 |
| [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py) | `Scheduler.schedule`（waiting 循环） | 43caf5b35 (#59030)、3e58641d1 (#60306) | encoder/lookahead 导致 0 token 时 skip；LoRA 计数取全部 running。4.9 流程图、4.9.1 注记并入正文。 |
| [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py) | `_try_schedule_encoder_inputs`、`_free_encoder_inputs`、`_ec_transfer_pending` | 12b005eef (#52412)、1641765ff (#60553)、ddc3f417d (#62834) | `<=` 判定与 encoder-decoder 排除；去掉 drafter 前瞻推后量；EC 回放。新增 4.16.1（公式、算例、流程图）与 4.16.2。 |
| [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py)、[vllm/v1/core/sched/output.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/output.py) | `_drafts_unused`、`SchedulerOutput.skip_drafts` | 856b98415 (#62784) | 新增 4.17.3（公式、符号、算例、流程图）。 |
| [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py)、[vllm/v1/core/sched/output.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/output.py) | `_get_encoder_input_updates`、`SchedulerOutput.encoder_input_updates` | 373614f66 (#57368) | 新增 4.17.4（规则、公式、算例）；4.17.2 表增两行。 |
| [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py) | `_update_request_with_output`、`update_from_output` | d486219fc (#58668)、f98ae6a9f (#58778)、13058ef4d (#62872) | #58668：stop 裁剪账目公式（4.19.3）；#58778：`continuing_reqs`；#62872：grammar 编译失败 `validation_error`（4.19.4）；4.19 流程图更新。 |
| [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py)、[vllm/v1/core/sched/async_scheduler.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/async_scheduler.py) | `_update_request_with_output`、`cache_blocks` 调用 | 2fb320d83 (#61589) | 删除 `AsyncScheduler._update_request_with_output`；placeholder 扣减与 `cache_blocks` 并入基类 `Scheduler`（4.19.3）。 |
| [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py) | `defer_async_loads`、`_drain_deferred_loads` | 8780c3f83 (#63827) | 新增 4.20.1（含时序图）。 |
| [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py)、[vllm/config/scheduler.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/config/scheduler.py)、[vllm/v1/engine/core.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/core.py) | `local_prefill_interval`、`prefill_schedule_interval` | ed3050add (#54627) | 4.21.2 重写（三种形态表、判定公式、算例、流程图）；4.22.1 表更新。 |
| [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py) | `num_hash_lookahead_tokens`、`_get_local_prefix_cache_hit` | 2fb320d83 (#61589)、aea0513a3 (#62591) | 4.15 补 lookahead 哈希、命中上限；公式与例子引向第 05 章。 |
| [vllm/v1/core/sched/interface.py](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/interface.py) | `SchedulerInterface.has_request` | d5b36699b (#51488) | 4.21.1 补充：引擎先拒绝再入 Scheduler。 |

## 不修改正文的 v0.33.0 变化

- `Scheduler.__init__` 中 EAGLE block drop 警告的触发条件改为 `needs_eagle_block_drop() and disable_eagle_block_drop`：只影响日志，不改调度语义。
- aux output（routed experts）的 block 哈希只发已计算或已调度块（#62489）：属于 aux output 连接器内部实现，本章只在 4.15 提到命中上限。
- `SchedulerConfig.long_prefill_token_threshold` 等其余参数未变。

## 对其他批次的提示

- 第 08 章（B2）需同步 `AsyncScheduler` 不再覆盖 `_update_request_with_output`：这来自 #61589（`2fb320d83`），不是 #58668（见 4.19.3）。
- 第 06/07 章（B5/B6）：`SchedulerOutput` 新增 `encoder_input_updates`、`skip_drafts`，Worker/Model Runner 侧语义由对应章节说明。
