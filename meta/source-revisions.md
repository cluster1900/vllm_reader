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
