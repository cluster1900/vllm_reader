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
| ch09 | §9.9.3、§9.7.7、Elastic EP 段、章末基线说明 | Elastic EP 与 MRV2 不支持清单的表述改为 `_get_model_runner_unsupported_features`（只剩 external_launcher+`PP>1` 与 DBO） | — |
| ch02、ch10 | 行号 | 仅机械刷新已验证的行号（ch10 另见下文第五节的基线文字改动） | — |
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

## 五、r2（评审后修订）

评审对 `308703bd` 提出 6 项必改与若干建议；修订前 `origin/main` 仍是 `c18e5124`，没有合并。

1. **ch07 §7.9 / ch05 §5.6.1（#63778）**：Worker 只按平台（`is_cuda_alike()` / `is_xpu()`）和 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 决定要不要估算（`gpu_worker.py` L748–L760，估算判断在 L755–L760）；`cudagraph_mode` / `cudagraph_mm_encoder` 的判断已由 `GPUModelRunner.profile_cudagraph_memory()` 承担，开启 `enable_extensible_kv_cache` 时返回 0（`gpu/model_runner.py` L1129–L1136，仅按评审给出的条件记录，机制留给后续批次）；其后由 `cudagraph_utils.py::profile_cudagraph_memory`（L1010–L1014）和 `needs_cudagraph_capture()`（`gpu/model_runner.py` L1138–L1146）判断。删去 7.9 “未发版”注记里的 #63778 条目，并把 7.32、8.26 的“仍只以注记保留”清单里的 #63778 去掉。
2. **ch07 行号**：`initialize_from_config` 调用 L394 → L408；`get_simple_cpu_offload_num_blocks` L402 → L432（`vllm/v1/engine/core.py`）。
3. **ch09 / ch10 章末基线说明**：原写“源码基线 4ba787ec / 已按 v0.33.0 静态复核”，改为 05caae03 基线，并如实说明引用与行号已复核、MRV1 删除已重绑，其余行为没有在 05caae03 上逐一重新核实。本批因此也有 ch09、ch10 的基线文字改动（此前误写为 ch10 只刷新行号）。
4. **ch05 §5.15.5 / §5.12.4**：按 `VllmConfig._resolve_num_hash_lookahead_tokens`（`vllm/config/vllm.py:632-653`）与 connector 的 `supports_lookahead_block_hashes`（#64096，`75be138428`）重写；connector 清单补全为 Offloading、SimpleCPUOffload、NixlBase、Mooncake、MooncakeStore 直接返回真，`MultiConnector` 逐个询问子 connector；§5.12.4 标题与读图方法去掉“(v0.33.0)”。
5. **读者正文里的流程标签 “B6/B8”**：已删除（README、ch01/03/04/05/06/07/08 的钉扎说明、`chapter-source-map.md`、ch09 末的“第二波 B8”、`source-revisions.md` 阅读基线 005）。已 grep 全书，剩余的 `B0`–`B8` 都是图中节点名（如 `B7[block 7]`）或实验里的 Baseline 记号，不是流程标签；`meta/source-revisions.md` 阅读基线 004 里更早的 “B0/B2/B3/B4” 属于旧 meta 历史记录，未改。
6. **“main 上的变化（未发版）”注记**：ch02、ch05–ch10 共 17 条注记 + ch07/ch08 各 1 条导语，前言改写为“已包含在本书基线中，正文尚未展开（后续批次补写）”，基线改为 `05caae03`。逐条用 `git merge-base --is-ancestor <提交> 05caae03` 核对（含 PR 号反查到提交），**48 个条目全部是 `05caae03` 的祖先，没有非祖先条目**（也都不在 v0.33.0 中，所以仍是“v0.33.0 之后”的变化）。因此没有任何注记保留“未发版”标签；导语写明“目前没有这样的条目”。清单：

| 位置 | 条目（PR / 提交） | 数量 |
|---|---|---|
| ch02 L345 | #64097（`9fc019a74`）、#64177（`f60edccd0`）、#63896（`5c1939140`） | 3 |
| ch05 L689 | #61294（`edad8b03c`）、#62860（`317081fb6`） | 2 |
| ch05 L1412 | #57646（`c13d4a5a4`）、#63781（`963399729`）、#60469（`e3883b7a7`）、#63889（`36f5ff317`）、#62592（`70cb50dab`）、#63779（`0beab6c25`）、#64229（`77f58e530`）、#63207（`f4fd1ab68`）、#64250（`a1cfd2c1b`）、#64230（`066d97993`）、#63513（`f79a3a7d4`）、#61793（`6cfb375fd`）、#64400（`55b677b2d`）、#59099（`4440f0e26`）、#63618（`415839510`）、#62930（`4a8c0da9e`）、#63566（`482e47325`）、#63788（`a57d25617`）、#64251（`491067f44`）、#59322（`d0ab3ba9b`）、#63937（`429eb2278`）、#64501（`8b286ac2b`）、#64345（`3c39dda2a`）、#62444（`c1b7dffa1`）、#64256（`16777da34`）、#64224（`fed7de909`）、#52838（`fa3b9d137`）、#63047（`7ab756940`）、#62860（`317081fb6`） | 29 |
| ch06 L445 | #63633（`78e2f7960`） | 1 |
| ch06 L1092 | #64064（`a6c3635e3`） | 1 |
| ch07 L718 | #61258（`4d95f222e`） | 1 |
| ch07 L820 | #63633（`78e2f7960`） | 1 |
| ch07 L1131 | #63219（`322f90a2a`） | 1 |
| ch07 L1299 | #62748（`d7f68d4c7`）、#59341（`06e172b5c`）、#63237（`65faf4efa`） | 3 |
| ch08 L476 | #63219（`322f90a2a`） | 1 |
| ch08 L860 | #64279（`d1ad431de`） | 1 |
| ch08 L1117 | #51012（`d6fbd9478`） | 1 |
| ch08 L1379 | #61464（`cdd2750cd`）、#62748（`d7f68d4c7`）、#63819（`458334f00`） | 3 |
| ch09 L269 | #61258（`4d95f222e`） | 1 |
| ch09 L758 | #63554（`26cbb47d4`）、#64356（`66875d312`）、#64151（`820be160c`）、#61464（`cdd2750cd`）、#61294（`edad8b03c`） | 5 |
| ch09 L1193 | #61994（`af6665b5d`） | 1 |
| ch10 L676 | #61258（`4d95f222e`） | 1 |

   去重后 48 个 PR/提交（ch05 L689 与 L1412 共有 #62860，ch07/ch09/ch10 共有 #61258，等等）；表中“数量”是各注记内的条目数。#42129 只作为 #64279 的回退对象被提及，不是条目。
7. **建议项**：ch06 `default.py` 行号改为 `:178`（FULL 分支）、`:180-181`（padded）、`:184-185`（unpadded）；ch01 `sampler.py:238` → `:239`；ch03 `release_kv_cache_memory` 补一句 `reset_connector` 参数（`core.py:1189`、`:1206`）。§8.14.3 `_decode_descs` 没有动，按评审说明跳过。

### 本批没有写入正文的上游漂移（评审裁定：留待后续批次）

以下提交都是 `05caae03` 的祖先；各章钉扎说明已加半句“行为未在 05caae03 上全面重核，见 meta note 中的清单”。

- #56492 extensible KV cache（`860da3daf`）
- #54625 cache-aware admission ordering（`2a777ce26`）
- #65285 / #65319 parked async KV load（`14d3ec2b8`、`56e993a68`）
- #60989 AR speculator 重构与 draft model CUDA Graph（`f352490fc`）

上游 `docs/` 里仍有 V1 旧文（含 `docs/design/model_runner_v2.md` 的 MRV1 对照），只记在这份 note 里，没有改动，也没有对外提交。“MRV2 没有对应 compile_sizes padding 校验”维持现状。
