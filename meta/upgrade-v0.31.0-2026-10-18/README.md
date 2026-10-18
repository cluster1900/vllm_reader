# dot：[TASK] #7 十章 v0.31.0 升级复核

- Reader 原始提交：`7d4cd098cf6ec1abd8c7b570dc1df5238cf37eef`
- 本报告源码：`db9527a46873454610df6dbedf79a36d6bf1a7f6`
- 日期：2026-10-18
- 章节状态仍为 draft / runtime_verified=false。

| 章 | 结论 | 问题/依据/对应文件 |
|---|---|---|
| 01 | 已更新（有实质影响） | [逐章报告](01-transformer-inference.md)；`book/01-transformer-inference/README.md` |
| 02 | 已更新（有实质影响） | [逐章报告](02-vllm-architecture.md)；`book/02-vllm-architecture/README.md` |
| 03 | 已更新（有实质影响） | [逐章报告](03-engine-core.md)；`book/03-engine-core/README.md` |
| 04 | 已更新（有实质影响） | [逐章报告](04-scheduler.md)；`book/04-scheduler/README.md` |
| 05 | 已更新（有实质影响） | [逐章报告](05-kv-cache.md)；`book/05-kv-cache/README.md` |
| 06 | 已更新（有实质影响） | [逐章报告](06-paged-attention.md)；`book/06-paged-attention/README.md` |
| 07 | 已更新（有实质影响） | [逐章报告](07-model-execution.md)；`book/07-model-execution/README.md` |
| 08 | 已更新（有实质影响） | [逐章报告](08-compile-cuda-graph.md)；`book/08-compile-cuda-graph/README.md` |
| 09 | 已更新（有实质影响） | [逐章报告](09-distributed.md)；`book/09-distributed/README.md` |
| 10 | 已更新（有实质影响） | [逐章报告](10-experiments.md)；`book/10-experiments/README.md` |

升级同时更新 README 基线、源码映射 JSON/Markdown、source-revisions 记录，以及 PCP+DP 教学模型与测试。前一基线的报告和实验记录保留为历史资料，不做全仓库字符串替换。

## 验证结果与保留边界

- 87 份 Reader 文本文件按原始 Git blob SHA 校验一致；源码文件按固定 revision 的递归 tree/blob SHA 校验，不把未校验的本地副本当作源码证据。
- 每个版本运行 `python3 -m unittest discover -s tests`：共 164 项，160 通过，4 项因未安装项目 Node 数学渲染依赖而跳过。
- 每个版本全部 11 个 `examples/ch*.py` 退出 0；`python3 scripts/enrich_pedagogy.py --check` 显示 0 章需补写。
- snapshot-aware 检查：章节元信息、围栏/空白、教学图注、公式、主源码符号与正文源码路径通过；不伪造 git HEAD。所选上游源文件快照中旧版 161、新版 165 个已下载文件与 Git blob SHA 一致。
- 十章 Markdown 已通过 Pandoc AST 解析。Mermaid 全量渲染、公式图像与 EPUB/网页视觉检查未运行；历史报告的 294 张图/462 张图片通过结果不被挪用为本次结果。
- 原样 `python3 scripts/check_book.py` 在本次快照环境不能完成：它要求同级完整 `../vllm` Git checkout，且本次未下载未修改的二进制封面。正式 checkout 中必须重新运行；这不是把源码符号检查失败当作通过。
- 未执行 vLLM 真模型、GPU kernel、CUDA Graph capture/replay、NCCL/Ray、多机、在线 SSE 或性能 benchmark。教学程序通过不构成上述运行正确性证明。

## 2026-10-08 全文报告未闭环项

旧报告保留边界仍有效：真实 NVIDIA 模型与多机实验需要固定硬件/模型/依赖/命令和实际输出；EPUB 在最终 Reader commit 合并后重建，以固定正确外链；网页部署和不同阅读器显示需另验。本次关闭的是列出的文档事实/图示缺口，不宣称关闭运行验证或发布检查。

## 审阅与合并

由 dot（OpenAI 助手，不是用户本人）准备，交 Normal Work 审核。所有变更只在 topic branch / draft PR；不直接推 main，不自动合并。源码上游保持只读。请按旧基线修正、再基线升级的顺序审核；两项 PR 分开。
