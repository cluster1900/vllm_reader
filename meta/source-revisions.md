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
