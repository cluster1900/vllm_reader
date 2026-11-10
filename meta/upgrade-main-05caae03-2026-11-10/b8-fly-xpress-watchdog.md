# B8a + watchdog 重钉 + B6 非阻塞修订（钉 `05caae03`）

- 阅读基线：vLLM main `05caae03c20f69436ff7ec23ded339ea104f1a01`（提交日期 2026-11-09；书 main `efaf5042305ee28f2bb585f2eb1ce765bafe9e0f`）。
- 跟踪：`cluster1900/bot-hub#127`。
- 章节仍为 draft，`runtime_verified=false`；全部静态源码核对，没有跑 GPU。
- 本批只改 ch03（watchdog 行号与章末一句）、ch04 章末、ch06 章末、ch07、ch08（一处措辞）、对应 `chapter-source-map` 条目、B6 meta note，以及本文件。未改 README、`source-revisions.md`、ch02、ch05、ch09、ch10。

## 一、祖先关系

在上游仓库对本书基线运行 `git merge-base --is-ancestor <提交> 05caae03c20f69436ff7ec23ded339ea104f1a01`：

| 提交（PR） | 主题 | 是 `05caae03` 祖先？ | 是 v0.33.0（`4ba787ec`）祖先？ |
|---|---|---|---|
| `c1dc2e9cf992995953548c392f8c67db7283495b`（#53987） | FLy | 是 | 是 |
| `ce9467b757c024a55fea4d0919ac2e682f28a894`（#54448） | XPress | 是 | 是 |
| `44e70e086ad6ae6a0726d3d830c9c8eee5db884b`（#65175） | XPress + Acceptance Estimator | 是 | **否** |
| `1c8799197`（#55700） | Engine/Worker watchdog | 是 | 是 |

#65175 已写入 7.22.1（Jacobi 把 logits 交给 acceptance estimator 的交接）；它不进 7.32 的 v0.33.0 核验表，只在表后说明。

## 二、B8a：ch07 §7.22.1 XPress 与 FLy

在 7.22（`init_speculator` 叙述）之后、7.23 之前新增短节。行号均在 `05caae03` 上手工核对：

- FLy：`compute_fly_entropy`（`fly.py:12`）；`rejection_sample`（`rejection_sampler_utils.py:1125`）在 `fly_window_size>0` 时以 `from_logits=True` 调用（`:1213-1218`）；内核推迟分支约 `:726-744`。配置与校验在 `speculative.py:628`、`:722-730`、`:1756-1776`。熵公式按实现写：top-$k$ 对全词表 softmax 的香农熵贡献，不重新归一化；附白话框。图带「读图方法」。
- XPress：`XPressSpeculator` / `_generate_draft` / `_jacobi_refine`（`xpress/speculator.py:16`、`:60`、`:82`）；`XPressSpeculatorConfig`（`config.py:7`）；`init_speculator`（`__init__.py:18`）。正文停在 speculator 层，不展开 `kernels.py`。`_CONFIG_BY_METHOD` 没有单独的 `"xpress"` 项；draft 经 `speculator_config_class` 声明，由 `get_speculator_config_class`（`:363`）在声明类是 method 配置类子类时采用。
- #65175：`return_logits=self.acceptance_estimator is not None`（`:98`）与 `_maybe_predict_acceptance`（`:104-108`），只记交接。

7.30 源码索引与 7.32 核验表已补 #53987 / #54448；`chapter-source-map.json` 为 ch07 增加上述锚点。

## 三、Watchdog：仅重钉行号（不新开章节）

书中已有完整短节：ch03 §3.10.4、ch07 §7.4.3。本批不新写机制。

在 `05caae03` 上复核：

| 符号 | 行号 |
|---|---|
| `NoopWatchDog` / `ThreadedWatchDog` | `watch_dog.py:29` / `:41` |
| `feed` / `disable` / `handle_failure` | `:64` / `:68` / `:76` |
| `get_watch_dog` / `start_watch_dog` | `:147` / `:152` |
| `WatchdogConfig` | `watchdog_config.py:13` |
| Engine 启动 / feed / disable / SIGTERM `handle_failure` | `core.py:1757` / `:1872` / `:1854` / `:1718` |
| Worker 启动 | `multiproc_executor.py:1081` |

ch07 §7.4.3 的启动点行号（Engine `core.py:1757`、Worker `multiproc_executor.py:1081`）在 B6 合入后已经正确，本批复核后未改正文。ch03 §3.10.4 原先没有行号，已按上表补上，并补 `WorkerProc` 启动锚点。`4ba787ec..05caae03` 上 `watch_dog.py` 仅有 docstring 反引号改动（`9b9bd049a`），`watchdog_config.py` 无行为 diff。

## 四、B6 评审非阻塞修订

1. ch07 §7.9 源码标签：整段是 `gpu_worker.py` L747–L760，是否估算的判断在 L755–L760。原标签写 L748–L760，已改。
2. `b6-mrv1-rebind.md` 文首「约定」和第四节有一条仍写「main 上的变化注记保留 v0.33.0 时代口径」，与第五节第 6 项（注记已改写、不再使用「未发版」标签）不一致。已把这两处改成与第五节一致：7.32 / 8.26 表仍保留 v0.33.0 口径；「main 上的变化」注记以 r2 为准。合并后的 PR #15 正文里，首段检查仍写 1197（r2 终值是 1202），以及同一条「约定」，另行用 `gh pr edit 15 --body-file` 只改这两处。
3. ch08 §8.17.1：原文写 `profile_cudagraph_memory`「整个包在 `eplb.suppress()` 里」。实际上 `enable_extensible_kv_cache` 时的 `return 0`（`gpu/model_runner.py:1131-1133`）发生在 `suppress()` 之前。已改措辞。
4. ch03、ch04、ch06 章末「已对照 v0.33.0 静态复核」补一句：那是 v0.33.0 时代的核对，当前阅读基线是 `05caae03`。ch05 由并行批次处理，本批未改。

## 五、未写入 / 停笔处

- XPress 的 Jacobi / top-C 内核（`xpress/kernels.py`）不展开。
- Acceptance estimator 本身不展开，只记 #65175 的交接。
- 未与 DFlash / DSpark 做对照表。
- Watchdog 超时公式的白话框维持原样，不新开章节。
- ch03 §3.4.4 原写「Elastic EP 的完整流程不在本章展开」，已改为「不在本章展开，见 9.8.6」。§9.8.6 由并行 PR #17 新增，本批只加交叉引用；#17 现已合入。
- ch07 §7.9 写 `enable_extensible_kv_cache` 时 `profile_cudagraph_memory` 返回 0，已加「见 5.6.5」。§5.6.5 由并行 PR #18 新增；§7.10 未提及 extensible KV，没有加引用。#18 现已合入。
