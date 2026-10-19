# dot：v0.32.0 十章基线升级（bot-hub #20）

- Reader base：`5f78d105690c1968885abb8c0264cd98fb7b9a18`。
- 旧源码：`db9527a46873454610df6dbedf79a36d6bf1a7f6`（v0.31.0）。
- 新源码：`5df63a7203cdb9328d824a27b682e77c7b9928be`（v0.32.0），tag ref 已核对；官方发布时间 2026-10-18 06:50:48 UTC。
- 日期：2026-10-19。章节保持 draft / runtime_verified=false。
- 任务：[bot-hub #20](https://github.com/cluster1900/bot-hub/issues/20)。

## 十章结论

| 章 | 结论 | 证据与无影响范围 |
|---|---|---|
| 01-transformer-inference | 已更新（有实质影响） | [逐章报告](01-transformer-inference.md)；不受变化影响的教学模型范围在报告中单独说明 |
| 02-vllm-architecture | 已更新（有实质影响） | [逐章报告](02-vllm-architecture.md)；不受变化影响的教学模型范围在报告中单独说明 |
| 03-engine-core | 已更新（有实质影响） | [逐章报告](03-engine-core.md)；不受变化影响的教学模型范围在报告中单独说明 |
| 04-scheduler | 已更新（有实质影响） | [逐章报告](04-scheduler.md)；不受变化影响的教学模型范围在报告中单独说明 |
| 05-kv-cache | 已更新（有实质影响） | [逐章报告](05-kv-cache.md)；不受变化影响的教学模型范围在报告中单独说明 |
| 06-paged-attention | 已更新（有实质影响） | [逐章报告](06-paged-attention.md)；不受变化影响的教学模型范围在报告中单独说明 |
| 07-model-execution | 已更新（有实质影响） | [逐章报告](07-model-execution.md)；不受变化影响的教学模型范围在报告中单独说明 |
| 08-compile-cuda-graph | 已更新（有实质影响） | [逐章报告](08-compile-cuda-graph.md)；不受变化影响的教学模型范围在报告中单独说明 |
| 09-distributed | 已更新（有实质影响） | [逐章报告](09-distributed.md)；不受变化影响的教学模型范围在报告中单独说明 |
| 10-experiments | 已更新（有实质影响） | [逐章报告](10-experiments.md)；不受变化影响的教学模型范围在报告中单独说明 |

## 删除/迁移符号闭环

- offline_utils 模块删除，但 OfflineInferenceMixin 及 `_run_completion` / `_run_engine` / `_add_request` 迁至 `vllm/entrypoints/common/offline.py`，并非离线 API 删除。
- coordinator 的 `get_replay_boundaries` 方法删除，改用模块纯函数 `replay_boundaries` 与 `get_num_cacheable_tokens` 边界。
- `BlockTable.map_to_kernel_blocks` 删除；持久表保持 manager IDs，`AttentionGroup.map_to_kernel_block_table` 在 backend metadata 边界转换。
- 原 DCP smaller-kernel-block 用例改为 `test_dcp_slot_mapping_only_shards_sharded_groups`；不再测试持久表展开。
- cascade attention 执行移除；旧兼容字段不代表功能仍可达。torch.compile SP/AsyncTP passes 移除不等于模型级 MoE SP 删除。
- JSON/Markdown 源码地图同步更新，旧基线记录和旧实验输出保留，不做全仓库版本字符串替换。

## 发布摘要与源码冲突

任务材料/release 摘要称共享内存 tensor arena 默认启用，但固定 tag 的
[vllm/config/parallel.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/parallel.py#L243-L250)
定义 `enable_shm_tensor_arena: bool = False`，并写明 experimental、opt-in 和 node-local 条件。
EngineArgs 继承此默认值，MultiprocExecutor 将配置值传入队列。本次按源码写明默认关闭，
保留差异供审阅，不把 release 摘要当成默认运行事实。

## 本次实际验证

- `python -m unittest discover -s tests`：174 项全部通过，0 skipped。已按锁文件安装本仓库 Node 工具，禁用安装脚本，四项数学渲染测试本次实际运行，未沿用旧报告的跳过结果。
- 逐一运行 11 个 `examples/ch*.py`：全部退出 0。
- `python scripts/enrich_pedagogy.py --check`：0 章需补写。
- 十章 Pandoc Markdown→JSON AST 解析通过；`git diff --check` 通过。
- 源快照专项检查：10 章、194 个主源码锚点、589 个符号、872 条显式/内联源码路径引用通过；172 份下载源码文件均与完整、未截断 immutable tree 的 Git blob SHA 匹配。清单见 [source-manifest.json](source-manifest.json)。这不是完整 Git checkout。
- [逐文件差异清单](mapped-file-changes.md)覆盖输入材料所列十章映射文件。文件发生变化与章节受影响范围分别判断，不能仅凭 diff 行数得出语义结论。

## 未通过或未执行的阶段

- 原样 `python scripts/check_book.py` 仍受当前快照环境限制：要求同级完整 `../vllm` Git checkout，且本次未物化未修改的二进制封面。上述快照检查不伪造 Git HEAD，也不等同于原样检查通过。正式 checkout 中应重跑该命令。
- 13 个已变更 Mermaid 图已提取并尝试 Chromium 渲染，但沙箱禁止 Chromium 创建所需 socket；授权执行路线也发生沙箱基础设施错误。图像渲染未完成，不能宣称视觉 QA 通过。
- 未构建/发布 EPUB 或网页；未做完整图像/排版视觉检查。最终合并 revision 应重新构建发布物。
- 未运行 vLLM 真模型、上游 GPU pytest、CUDA Graph capture/replay、NCCL/Ray、多机、线上 SSE 或性能 benchmark。CPU 教学测试不证明这些运行路径。

## 审阅与合并

由 dot（OpenAI 助手，不是用户本人）准备。只交付主题分支和 Draft PR，
不直接推 main、不自动合并；由 Normal Work 独立复核后决定合并与发布。
重点复核 max_tokens 的入口/输出/context 三边界、manager/kernel 坐标、PCP decode owner 和 release 摘要冲突。
