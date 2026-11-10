# 源码版本记录

## 阅读基线 001

- Date: 2026-09-07
- Repository: `vllm-project/vllm`
- Local path: `/Users/hawk_wu/Desktop/vllm`
- Branch: `main`
- Commit: `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`
- Dirty: `false`
- Commit subject: `[Bugfix] DSv4 MXFP4 selector: stop narrowing explicit aliases to their BF16 variant (#53586)`
- Purpose: 建立全书结构并核对 V1 Engine、Scheduler、KV Cache、Model Runner、
  CUDA Graph 和分布式执行的当前模块边界。

## 更新规则

新增阅读基线时，不覆盖旧记录。正式章节必须记录其使用的 commit；如果章节跨越
多个 revision，应说明每个 revision 用于验证什么结论。

## 基线 001 复查：2026-10-08

- 实际源码路径：`../vllm`，已核对为本项目同级仓库。
- Branch / commit / dirty：`main` / `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b` / `false`。
- 未更新、切换或修改上游源码；本轮仍复核基线 001。
- 覆盖范围和结论见[文档复核记录](full-document-review-2026-10-08.md)。

## 阅读基线 002：v0.31.0（2026-10-18 静态复核）

- Repository: `vllm-project/vllm`
- Tag: `v0.31.0`
- Commit: `db9527a46873454610df6dbedf79a36d6bf1a7f6`
- 前一基线：`5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`；保留上方历史记录。
- Reader base: `7d4cd098cf6ec1abd8c7b570dc1df5238cf37eef`。
- 获取方式：GitHub 连接器按完整 commit 只读获取源码；逐文件 Git blob SHA 与对应 immutable recursive tree 一致。是选定文件快照，不宣称完整 git checkout 或作者电脑工作树状态。
- source_path `../vllm` 是读者复现时的约定；本次未访问或切换用户电脑的上游仓库。
- 检查范围：十章源码、图示与引用的静态升级，CPU 教学脚本/单元测试；不是上游 CUDA/NCCL/真实模型验证。
- 逐章“已更新/无影响”结论及证据：[升级复核](upgrade-v0.31.0-2026-10-18/README.md)。
- 章节保持 `draft` / `runtime_verified: false`。历史实验输出不被改写为新基线运行结果；EPUB/网页发布仍须重建并验证。


## 阅读基线 003：v0.32.0（2026-10-19 静态复核）

- Repository: `vllm-project/vllm`
- Tag: `v0.32.0`
- Commit: `5df63a7203cdb9328d824a27b682e77c7b9928be`
- 前一基线：`db9527a46873454610df6dbedf79a36d6bf1a7f6`；旧记录和旧实验输出原样保留，不作为本次运行证据。
- Reader base: `5f78d105690c1968885abb8c0264cd98fb7b9a18`。
- 获取方式：连接器读取固定 commit 文件，并逐一验证 Git blob SHA 与未截断 recursive tree 一致。
- 本次是只读源码快照，不是完整上游 Git checkout；`../vllm` 是读者复现约定，不声称访问用户电脑或检查其 dirty 状态。
- 十章均给出影响与证据，见[升级复核](upgrade-v0.32.0-2026-10-19/README.md)；章节继续 `draft` / `runtime_verified: false`。
- Python 要求 `>=3.11,<3.15`。release 摘要的 tensor arena 默认开启说法与源码默认 False 不一致，以源码为准并保留差异记录。
- 验证仅限本次实际执行的 Reader 教学测试、脚本与静态检查；没有运行真模型、GPU、CUDA Graph、NCCL、多机或性能实验。

## 待升级清单：main 跟踪（2026-10-25，未发版）

- 当前基线仍为 003（`v0.32.0` / `5df63a7203cdb9328d824a27b682e77c7b9928be`）；不切换基线、不改写 v0.32.0 正文。
- 跟踪范围：vLLM `main` 2026-10-19～2026-10-25，截至 `a7407ab44`（#63616）。v0.32.0 位于 release 分支，与 main 的 merge-base 为 `fe341a53`。
- 逐条 SHA/PR、受影响章节与升级时动作见[main 跟踪记录](main-tracking-2026-10-25.md)；章节内以“main 上的变化（未发版）”注记标出。
- 下一个发版 tag 出现后，按该记录逐条复核并从注记转为正文，再新增阅读基线 004。

## 待升级清单：main 跟踪（2026-10-27，未发版）

- 当前基线仍为 003（`v0.32.0`）；不切换基线、不改写 v0.32.0 正文。
- 跟踪范围：vLLM `main` `a7407ab44`..`ed1a73079`（#63689，2026-10-27），101 个提交。
- 逐条 SHA/PR、受影响章节与升级动作见[main 跟踪记录](main-tracking-2026-10-27.md)；ch03、ch05、ch06、ch07、ch08、ch09 共 10 处加了“main 上的变化（未发版）”注记，其中 #61589、#63633、#62330 与 v0.32.0 正文冲突，升级时须改写正文。


## 待升级清单：main 跟踪（2026-10-28，未发版）

- 当前基线仍为 003（`v0.32.0`）；不切换基线、不改写 v0.32.0 正文。
- 跟踪范围：vLLM `main` `ed1a7307950ab5e6593d25123579ed7ca998b350`..`0106c8abce0f8fec21d2871e5b7404acd9c75065`（#61831，2026-10-28），103 个提交。
- 逐条 SHA/PR、受影响章节与升级动作见[main 跟踪记录](main-tracking-2026-10-28.md)；ch05、ch07、ch08、ch09 加了“main 上的变化（未发版）”注记，其中 #60960、#64151、#63778、#61258 与 v0.32.0 正文冲突，升级时须改写正文。

## 待升级清单：main 跟踪（2026-10-29，未发版）

- 当前基线仍为 003（`v0.32.0`）；不切换基线、不改写 v0.32.0 正文。
- 跟踪范围：vLLM `main` `0106c8abce0f8fec21d2871e5b7404acd9c75065`..`36d4ef654a83e227216b0c0b55cf4041d96b1675`（#64387，2026-10-29），76 个提交。
- 逐条 SHA/PR、受影响章节与升级动作见[main 跟踪记录](main-tracking-2026-10-29.md)；ch02、ch05、ch07、ch08、ch09 加了“main 上的变化（未发版）”注记，其中 #61464 与 v0.32.0 正文冲突，升级时须改写正文。

## 待升级清单：main 跟踪（2026-10-30，未发版）

- 当前基线仍为 003（`v0.32.0`）；不切换基线、不改写 v0.32.0 正文。
- 跟踪范围：vLLM `main` `36d4ef654a83e227216b0c0b55cf4041d96b1675`..`2eae56c4ba0cf12087fec035514eedede83c257d`（#64610，2026-10-30），109 个提交。最新 GitHub Release 仍为 v0.32.0（2026-10-18）。git rc tag 有 `v0.32.1rc0`（2365177dd，2026-10-18，在 main 上、早于本区间起点）、`v0.33.0rc0`（4135590ff，2026-10-26，在 main 上、早于区间起点）、`v0.33.0rc1`（cbafc4690，2026-10-28，不在 main 历史上，位于发布分支）；三者都没有 GitHub Release 条目（连 pre-release 也没有），不是阅读基线。
- 逐条 SHA/PR、受影响章节与升级动作见[main 跟踪记录](main-tracking-2026-10-30.md)；已在 ch02、ch05、ch06、ch07、ch08、ch09 加了“main 上的变化（未发版）”注记。#64279：v0.32.0 基线自身带 #42129 allow-list，Transformers backend + 整图编译会 KV 损坏。#64356：ch09 中 #63554 的 owner/row 规则已过时。本区间没有与 v0.32.0 正文直接冲突、升级时必须改写的提交。

## 阅读基线 004：v0.33.0（2026-11-10 B0 钉扎）

- Repository: `vllm-project/vllm`
- Tag: `v0.33.0`（annotated tag 解引用 commit；tagger date **2026-11-01** 09:21:10 UTC；GitHub Release 名为「v0.33.0」的条目 `tag_name` 为 `v0.33.0rc3`，指向同一 commit，published 2026-11-02）
- Commit: `4ba787ec9959e566d3b587a796dff07673324da5`
- 前一基线：`5df63a7203cdb9328d824a27b682e77c7b9928be`（v0.32.0 / 基线 003）；旧记录、旧实验输出与 `meta/main-tracking-*.md` 注记原样保留。
- Reader base：本 B0 PR 合入前的 `main`；本批钉扎 README / source-revisions / chapter-source-map / 十章 `source_commit`+`source_branch`，并把 check_book 的 6 处 map 失败改到 v0.33.0 实址（见下）；**不改写章节正文**（十章 `verified_at` 仍为 2026-10-19）。
- 获取方式：同级 `../vllm` 以 detached HEAD 检出上述 commit（由 annotated tag `v0.33.0` 解引用核对）。
- Python 要求仍为 `>=3.11,<3.15`（`pyproject.toml` `requires-python`）。
- B0 map 再绑定（仅 JSON/md，正文留给后续批）：
  - ch05：`check_enough_kv_cache_memory` / `get_kv_cache_config_from_groups` / `get_kv_cache_groups` / `get_kv_cache_configs` → `vllm/v1/core/kv_cache_planning.py`（正文仍提旧路径 → **B1**）
  - ch08：`_update_request_with_output` → `vllm/v1/core/sched/scheduler.py`（**属 ch08，正文批 B2**，不是 B4/ch04）
  - ch09：删除 `ray_executor.py` / `RayDistributedExecutor` 条目（类已不存在；既有 `ray_executor_v2.py` / `RayExecutorV2` 覆盖 → 正文 **B3**）
- 正文落注记按 bot-hub #127：波次 1 仅落 **v0.33.0 ancestry** 内变更；非 ancestry 注记推迟后续波次。
- 章节继续 `draft` / `runtime_verified: false`。


## 阅读基线 005：vLLM main `05caae03`（2026-11-10 重新钉扎）

- Repository: `vllm-project/vllm`
- Branch: `main`（未打 tag 的 commit；最近的 tag 是 `v0.34.0rc0` = `34b780fe66bdf3df173358774eb4f29f684b8368`，2026-11-08，`05caae03` 在它之后 112 个提交，`git describe` 为 `v0.34.0rc0-112-g05caae03c`）
- Commit: `05caae03c20f69436ff7ec23ded339ea104f1a01`（2026-11-09 16:54:29 -0800 = 2026-11-10 08:54 SGT；标题 “[CI] Fix B200 nightly gsm8k evals: ReplaySSM and DeepEP LL (#66207)”）
- 前一基线：`4ba787ec9959e566d3b587a796dff07673324da5`（v0.33.0 / 基线 004）。v0.33.0 在发布分支上，与 `main` 的 merge-base 是 `4135590ffe0119dab806d78a7eee6dff3be2618e`（v0.33.0rc0）。
- 这次重新钉扎的直接原因：`aac1b2a4f36e742c1f55bcb34ca9aaa5e6e02592`（#63373 “[Core] Remove Model Runner V1”，2026-11-07）。它是 `05caae03` 和 `v0.34.0rc0` 的祖先，**不在** v0.33.0 中；MRV1 的整套代码（`vllm/v1/worker/gpu_model_runner.py`、`v1/sample/sampler.py`、`v1/worker/block_table.py`、`v1/cudagraph_dispatcher.py` 等）在这个 revision 里都不存在。
- Python 要求仍为 `>=3.11,<3.15`（`pyproject.toml` `requires-python`）。
- 获取方式：同级 `../vllm` 检出上述 commit（detached HEAD）。
- 正文改写：这次重新钉扎把所有依赖 MRV1 的叙述改写为 MRV2 唯一路径（ch01 sampler、ch03、ch04、ch05、ch06、ch07、ch08、ch09），并顺带最小修正 check_book 报出的非 MRV1 漂移；行号引用已全部对 `05caae03` 手工核对。清单见 `meta/upgrade-main-05caae03-2026-11-10/b6-mrv1-rebind.md`。
- 保留为历史的内容：各章的 “main 上的变化（未发版）” 注记与 v0.33.0 祖先链核验表（ch07 7.32、ch08 8.26）保留，前言已改为“已包含在本书基线中，正文尚未展开（后续批次补写）”；逐条核对结果（17 条注记、48 个条目全部是 `05caae03` 的祖先）见 note。
- block_size=64、extensible KV、XPress/FLy、Elastic EP、`/v1/decisions`、watchdog 等新内容不在本批，只在正确引用所必需时提到；未写入正文的上游漂移清单见 note。
- 章节继续 `draft` / `runtime_verified: false`。

## 待升级清单：main 跟踪（2026-11-10，恢复；未发版）

- 当前基线仍为 005（main `05caae03`）；不切换基线、不改写章节正文。最新 GitHub Release 仍是 v0.33.0，最近 git tag 仍是 `v0.34.0rc0`，没有新 tag。
- 跟踪范围：vLLM `main` `05caae03c20f69436ff7ec23ded339ea104f1a01`..`db1a63c17812a14f14ec5f21bc0a33eaa5c31a25`（#65996，2026-11-10 10:32 SGT），8 个提交。这是第二波（B6–B8，#15–#18）合入后恢复的 main 跟踪，接续 [2026-10-30 记录](main-tracking-2026-10-30.md)。
- 逐条 SHA/PR、受影响章节与升级动作见[main 跟踪记录](main-tracking-2026-11-10.md)：区间内必须修正 0 条；另有 1 条区间外既有事实错误（ch06 §6.10：`compute_mm_prefix_ranges` 另有 `cpu/model_states/default.py:135` 调用），已在本改正文小批作为第一项修正；后续批次 4 条（#65996、#63432、#65983、#64715）；仅清单 4 条。`db1a63c17` 只是快照截止点，之后上游已有 #64403、#64001，归入下一个跟踪区间。
