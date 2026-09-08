# 写作路线图

状态定义：`outline`、`draft`、`verified`、`stale`。

| 章 | 主题 | 状态 | 主要前置 | 第一轮交付 |
|---|---|---|---|---|
| 01 | Transformer 与自回归推理 | draft | 无 | 最小推理器、prefill/decode 图 |
| 02 | vLLM 全景与请求生命周期 | draft | 01 | 端到端调用链、进程图 |
| 03 | Engine 与 EngineCore | draft | 02 | EngineCore step 与 busy loop |
| 04 | Scheduler 与 Continuous Batching | draft | 03 | token 级调度模拟器 |
| 05 | KV Cache | draft | 04 | block 生命周期模拟器 |
| 06 | PagedAttention | draft | 05 | block table 寻址图与 backend 路径 |
| 07 | Executor、Worker、Model Runner | draft | 03、04、06 | 单步模型执行调用链 |
| 08 | Compile 与 CUDA Graph | draft | 07 | eager/compile/graph 对照实验 |
| 09 | 分布式推理 | draft | 07 | TP/PP/DP/EP/PCP/DCP 拓扑、通信图与教学模拟器 |
| 10 | Benchmark 与优化 | draft | 08、09 | 指标分析器、可重复性能实验与回归报告 |

## 里程碑

- **M1：能解释一次请求。** 完成 01-03 草稿。
- **M2：能解释吞吐来源。** 完成 04-07 草稿。
- **M3：能解释生产执行。** 完成 08-09 草稿。
- **M4：能测量和优化。** 完成 10 草稿及全书实验复核。
- **M5：第一版发布。** 全章 verified，术语、图、引用和版本信息一致。

## 当前下一步

第 01 至 10 章正文、图示、最小实现、CPU 测试和实验模板已经完成。全书已完成面向
“有 Python 基础、缺少推理系统训练的初中级程序员和软件工程学生”读者的教学层复核：每章新增分层阅读
方法，所有直接进入代码、表格或子问题的小节都有阅读入口，全部 Mermaid 图都有读图
说明。EPUB 构建链路可将 Mermaid 渲染为 SVG 并做结构校验。真实 NVIDIA GPU trace 和
独立人工审阅仍待补，因此章节保持 draft 而不是 verified。

## 2026-09-08 全书源码与教学复核

十章完成本轮通读、重点源码对照与修复，详见
[检查报告](full-book-review-2026-09-08.md)。修复范围包括默认 client、调度计数、缓存
生命周期、执行调用者、graph fallback、并行限制、通信顺序与指标量纲；各章补充具体
阅读路线和小练习，术语表同步扩充。静态核对和 CPU 测试不替代 GPU 动态验证。


## 再次复核与交付

新增 [全书再次复核记录](final-source-review-2026-09-08.md)，修复教学程序的提交顺序和
取消/长度边界，收紧 pause、prefix lookup、UVA 与计时解释。第 05–10 章增加初读自检，
全局术语表补齐进阶缩写；定义引用按 AST 核对完整名称。封面与作者信息、公式排版及
Markdown/EPUB 全文一致性检查随本轮一起交付，章节保持 draft。
