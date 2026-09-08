# vLLM 源码教程（图文）

面向初中级程序员和软件工程类学生，具备 Python 基础即可，不要求具备大模型系统背景，系统讲解 vLLM V1 的
核心架构、源码调用链和性能优化机制。

本书不采用逐文件翻译的方式。每章都从一个可观察的问题出发，依次经过直觉模型、
最小实现、系统位置、真实源码、运行实验和设计取舍，最终让读者能够独立定位和
分析 vLLM 的新功能。

## 当前基线

- 上游仓库：`vllm-project/vllm`
- 分支：`main`
- Commit：`5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`
- 基线日期：2026-09-07
- 文档状态：第 01 至 10 章正文、图解、教学代码和实验模板均已完成
- 本轮全书复核：2026-09-08；修正源码语义、图示和教学示例，增加分层路线与小练习。
- [逐章检查与修复记录](meta/full-book-review-2026-09-08.md)：问题、证据、验证结果及未运行范围。

正式章节必须绑定自己的源码 commit。上游更新后，不得假设旧结论仍然有效。

## 全书目录

| 章 | 标题 | 目标篇幅 | 状态 |
|---|---|---:|---|
| 01 | [从 Transformer 到大模型推理](book/01-transformer-inference/) | 50 页 | draft |
| 02 | [vLLM 全景与一次请求的生命周期](book/02-vllm-architecture/) | 45 页 | draft |
| 03 | [Engine 与 EngineCore](book/03-engine-core/) | 45 页 | draft |
| 04 | [Scheduler 与 Continuous Batching](book/04-scheduler/) | 65 页 | draft |
| 05 | [KV Cache 内存管理](book/05-kv-cache/) | 75 页 | draft |
| 06 | [PagedAttention 与 Attention Backend](book/06-paged-attention/) | 65 页 | draft |
| 07 | [Executor、Worker 与 Model Runner](book/07-model-execution/) | 60 页 | draft |
| 08 | [异步执行、torch.compile 与 CUDA Graph](book/08-compile-cuda-graph/) | 45 页 | draft |
| 09 | [分布式与多 GPU 推理](book/09-distributed/) | 60 页 | draft |
| 10 | [实验、性能分析与优化](book/10-experiments/) | 50 页 | draft |

`draft` 表示章节正文、图解和无 GPU 实验已经形成，但真实 NVIDIA GPU 动态验证或独立
人工复核仍待完成；`outline` 表示源码入口已建立，正文尚在扩写。

## 构建 EPUB

项目使用 Mermaid CLI 将章节中的 Mermaid 图渲染为 SVG，再由 Pandoc 生成 EPUB3：

```bash
pnpm install
python3 scripts/build_epub.py
```

默认产物为 `dist/vllm-source-guide.epub`。构建结束时会自动运行
`scripts/check_epub.py`，检查 EPUB 容器、目录、正文、图片和内部链接是否完整。

全书还使用 `scripts/enrich_pedagogy.py` 维护教学提示。每章都包含针对目标读者的阅读
方法，每张 Mermaid 图后都有读图说明；`scripts/check_book.py` 会检查这些内容没有遗漏。

目标总篇幅约 560 页。页数是控制内容比例的估算，不是完成标准。

## 推荐阅读路线

- **快速建立主干：** 01 -> 02 -> 03 -> 04 -> 05 -> 06（先读寻址与元数据）-> 07 -> 10（先读指标与实验方法）。
- **深入 GPU 执行：** 主干完成后阅读 06 -> 08。
- **进入生产部署：** 主干完成后阅读 09 -> 10。
- **源码贡献准备：** 全部章节，并完成每章的源码追踪题和实验。

详细的章节依赖、写作契约和验收标准见 [全书写作蓝图](book/README.md)。

## 项目文件

- [AGENTS.md](AGENTS.md)：人类与 AI agent 的源码研究和写作规范。
- [全书写作蓝图](book/README.md)：章节关系、统一结构和阅读路线。
- [写作路线图](meta/roadmap.md)：各章进度和里程碑。
- [源码版本记录](meta/source-revisions.md)：所有研究基线。
- [章节源码映射](meta/chapter-source-map.md)：十章与当前实现的关键符号对应关系。
- [术语表](glossary/terms.md)：跨章节统一使用的术语定义。

## 结构校验

```bash
python3 scripts/check_book.py
```

该命令检查十章文件、frontmatter、内部链接、Markdown 代码围栏，以及
`meta/chapter-source-map.json` 中的源码路径、关键符号和上游 commit。它不替代对
控制流、边界条件和实验结果的人工复核。
