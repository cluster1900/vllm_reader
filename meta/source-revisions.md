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
