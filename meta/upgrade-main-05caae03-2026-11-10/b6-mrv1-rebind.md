# B6：基线重钉到 vLLM main `05caae03`，并改绑 MRV1 删除后的正文

- 结论：**基线从 v0.33.0（`4ba787ec9959e566d3b587a796dff07673324da5`）重钉到 vLLM main `05caae03c20f69436ff7ec23ded339ea104f1a01`**（提交日期 2026-11-10 08:54 SGT，`[CI] Fix B200 nightly gsm8k evals: ReplaySSM and DeepEP LL (#66207)`）；所有依赖 Model Runner V1 的叙述已改绑 MRV2 或明确标注“已删除”。
- Reader base：`c18e5124d10bff83405e52e3f5c530dd8148ccfb`。跟踪 issue：`cluster1900/bot-hub#127`。
- 触发变更：`aac1b2a4f36e742c1f55bcb34ca9aaa5e6e02592`（#63373，`[Core] Remove Model Runner V1`，2026-11-07）是 `05caae03` 与 `v0.34.0rc0` 的祖先，**不在** v0.33.0 里。删除后 `vllm/v1/worker/gpu_model_runner.py` 及其整套（`block_table.py`、`v1/sample/sampler.py`、`cudagraph_dispatcher.py`、`cp_utils` 里的 MRV1 符号等）不复存在，`vllm/v1/worker/gpu/…` 的 `GPUModelRunner` 是唯一 Runner。
- 删除之后的配置面（正文均已按此重写）：`VLLM_USE_V2_MODEL_RUNNER` 未设置 OK、值为 `1` 只 warning、其他值在 `Platform.validate_environ` 里 `ValueError`；`_validate_model_runner` / `_get_model_runner_unsupported_features` 只剩 external_launcher+`PP>1` 与 DBO 的组合；standalone CPU Runner 有自己的不支持清单；Worker 另有 Triton 检查。
- 章节仍为 draft，`runtime_verified=false`；全部静态源码核对，没有跑 GPU。
- 约定：各章里“main 上的变化（未发版）”小节和 7.32、8.26 的发布祖先表保留为 v0.33.0 时代的历史记录；只有与新正文**直接矛盾**的条目被合并或删除（#61464、#60960、#63778）。
- 写法约定：已删除的文件在正文里用短写（如 `v1/sample/sampler.py`），避免被 `check_book` 当成指向缺失文件的源码路径；历史对照引用 `docs/design/model_runner_v2.md`，并标明是历史对照。

## 一、MRV1 删除带来的改绑（按章）

| 章 | 位置 | 改绑内容 | 图/公式 |
|---|---|---|---|
| ch01 | §1.5 采样（Penalty/Bias、Top-k 等）、模型调用链、源码地图、证据矩阵、异常路径、索引 | MRV1 `Sampler`（`v1/sample/sampler.py`）已不存在，叙述改为 MRV2 `vllm/v1/worker/gpu/sample/sampler.py::Sampler` 的 logits 处理与采样顺序；`topk_topp_sampler.py` 说明为被复用的算子；`VLLM_USE_V2_MODEL_RUNNER` 排障条目改为“不再是开关”；调用链起点改为 `GPUModelRunner` | 新 mermaid（采样顺序流程图）、调用链图改 MRV2 |
| ch06 | §6.3.3、6.5、6.6、6.8、6.10、6.11、6.13、6.15.5、6.16 | 持久 `BlockTable`/`map_to_kernel_blocks` 的 MRV1 叙述删除，块表、gather、slot mapping、metadata 构造只讲 `vllm/v1/worker/gpu/block_table.py` 与 `attn_utils.py`；`get_block_table_width` 的新家为 `kv_cache_interface.py`；清掉 `num_common_prefix_blocks` 残留；删去一条 wave-1 里已不成立的“弃用隐式生成 CPU seq_lens”说法 | 新流程图（metadata 复用） |
| ch07 | 本章定位、7.2、7.3、7.4.3、7.5、7.6/7.6.2、7.9、7.10、7.12、7.13、7.15、7.16、7.18–7.22、7.25–7.28、7.32、小结 | “V1 Engine 必然 MRV1/按环境变量选 Runner”全部重写；7.2 改为“V1 Engine 与已被移除的 MRV1”；7.6.2 改为“Runner 什么时候构造”（`_make_model_runner` 只在 MM-encoder-only 与 `GPUModelRunner` 之间选）；7.12 persistent batch 只讲 MRV2；7.22 改为“选 MRV1 还是 MRV2 的规则已不存在”并指向 `Platform.validate_environ` / `_validate_model_runner`；7.19 grammar mask 汇合点改 MRV2；测试条目改为 `test_model_runner_supports_*`；7.32 之前的 main 注记保留 | 7.18、7.22 新流程图，7.6 时序图与流程图改 MRV2；7.16 状态图只加注 |
| ch08 | 本章定位、8.2.3、8.3.1、8.4.2、8.4.3、8.7.1、8.8.3、8.8.5、8.11、8.13.1、8.14.x、8.15.3、8.16.2、8.17、8.20、实验、练习、8.23 | 8.2.3 async 队列容量只剩“P，开启 async 则 P+1”；MRV1 + async + PP 的拒绝逻辑及 `v1_async_sched_incompat`（#62521）删除；8.11 `CudagraphDispatcher` 改为“已删除：职责现在在哪里”（保留编号，映射到 `CudaGraphManager.dispatch/capture`）；8.15.3 的 MRV1 动态 SD（#46399）删除；8.16 稳定态 MRV1 小节删除并重编号；实验 D 改为 MRV2 的 descriptor 与降级；compile_sizes padding 校验随 MRV1 消失 | 8.2.3 流程图重写；8.11 图保留（只改读图说明） |
| ch09 | §9.9.3、§9.7.7、Elastic EP 段 | Elastic EP 与 MRV2 不支持清单的表述改为 `_get_model_runner_unsupported_features`（只剩 external_launcher+`PP>1` 与 DBO） | — |
| ch02、ch10 | 行号 | 仅机械刷新已验证的行号 | — |
| README、book/README、meta | 基线行、版本政策、`chapter-source-map.json/.md`、`source-revisions.md` 阅读基线 005 | `source_commit` 改为 `05caae03`；地图里已不存在的符号（`gpu_model_runner.py`、`v1/sample/sampler.py`、`block_table.py`、`cudagraph_dispatcher.py`、`cp_utils` 符号、`compute_hash`、旧 v2 config 符号和对应测试）删除或替换；新增 `tests/v1/engine/test_engine_core.py`、`v1/request.py`、`get_block_table_width`、`validate_environ`、`CudaGraphManager.dispatch/capture`、`pcp_unsupported_reason`、`update_max_model_len` 等条目 | — |
| examples / experiments / scripts | `examples/ch07_execution_pipeline.py`、`examples/ch08_async_compile_graph.py`、`experiments/ch07-*.md`、`experiments/ch08-*.md`、`scripts/enrich_pedagogy.py` | 注释、docstring 与实验步骤去掉 MRV1/MRV2 对照；`enrich_pedagogy.py` 的一句话与 ch07 保持一致 | — |

## 二、行号机械刷新

先用 `git diff -U0 v0.33.0..05caae03` 的 hunk 把旧行号映射到新行号，只在“旧行与新行文本完全相同”时接受；其余全部人工重查。引用行与符号由 `check_book`（207 个源码锚点、1197 条显式引用）和符号校验脚本核对。

## 三、非 MRV1 的漂移修正（单列；只做最小改动）

| 章 | 内容 | 上游 |
|---|---|---|
| ch04 §4.21.2 | 预填充延迟（prefill interval）规则重写：`defer_prefills`、`prefill_capacity_bound`，新增公式、白话解释与流程图 | #66224 |
| ch04 §4.17.1 | 恢复的请求以 `NewRequestData` 发出；删掉 `num_common_prefix_blocks` 一行与源码行 | — |
| ch05 §5.12.1 | 前缀缓存命中上限改为 `prefix_cache_hit_limit`（合并并删除“main 上的变化”旧注记） | #60960 |
| ch05 §5.12.4 | lookahead hash 改用 `_resolve_num_hash_lookahead_tokens`，加入 connector 的 `supports_lookahead_block_hashes` 判断；BlockPool 重命名 | — |
| ch03 §3.6.3 | batch queue 容量 async 时为 P+1（`vllm.py:559-565`）；补 draft `max_model_len` 更新链 | — |
| ch06 §6.16.6 | `num_common_prefix_blocks` 清理；`common_prefix_len` 说明 | — |
| ch08 §8.4.3 | `cache_blocks` 的调用条件多出第三分支：Mamba + 同步调度 + prefix caching | — |
| ch08 §8.x | `compile_hash` → `compile_factors`；FULL/PIECEWISE 捕获顺序注释；capture-size 配置拆分与 `_decode_descs` 请求数表（只简短点到） | #65652 |
| ch09 §9.9.3 | 投机解码与 PCP 的限制改为 `pcp_unsupported_reason` | — |
| ch09 §9.7.7 | 新增 `_has_global_unfinished_reqs` 容错分支一段（`core.py:2864-2866`） | — |
| ch07（§7.25 附近） | `init_speculator` / `speculator_class` 命名 | — |
| ch08 §8.2.3、ch08 合并注记 | #63778 `use_v2_model_runner` 条件的合并说明 | #63778 |

## 四、存疑与未尽事项

- ch08 §8.14 的 capture-size 配置拆分（`prefill_/decode_cudagraph_capture_*`）和 `_decode_descs` 请求数表只做了简短提示，没有展开重写；如要完整覆盖需另起一批。
- MRV2 里没有对应 compile_sizes padding 校验，正文只写“已删除”，没有断言替代行为。
- 7.32、8.26 的发布祖先表与“main 上的变化”注记保留 v0.33.0 时代口径，不是遗漏。
- 上游 `docs/` 里仍有 V1 历史叙述（含 `docs/design/model_runner_v2.md` 的 MRV1 对照），本批未据此改动；按规则不对外提交，仅记在 bot-hub。
- B7（block_size=64、可扩展 KV）与 B8（XPress/FLy、Elastic EP、`/v1/decisions`、watchdog）内容未写；只在引用需要时点到。
