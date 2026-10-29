# vLLM main 跟踪记录（2026-10-28～2026-10-29，未发版）

- 当前阅读基线：`v0.32.0` / `5df63a7203cdb9328d824a27b682e77c7b9928be`（基线 003），正文不变。上游仍无新 release（最新 v0.32.0，2026-10-18）。
- 跟踪区间：`vllm-project/vllm` `main` `0106c8abce0f8fec21d2871e5b7404acd9c75065`..`36d4ef654a83e227216b0c0b55cf4041d96b1675`（[#64387](https://github.com/vllm-project/vllm/pull/64387)，2026-10-29 08:33 UTC+8），共 76 个提交。接续 [2026-10-28 记录](main-tracking-2026-10-28.md)。
- 方法：在只读 partial clone 中逐条 `git show` 阅读下表提交的非测试源码 diff；只记录读过 diff 的条目。没有运行上游代码、GPU 或模型。影响表共 28 行（重点领域），其余 CI/ROCm/XPU/recipe/文档/安全与模型边角在文末一句带过。行号均按 tip `36d4ef654` 核对。
- 结论：v0.32.0 正文对 v0.32.0 本身仍成立；共加/追加“main 上的变化（未发版）”注记：ch02（run-batch `/v1/completions` #64097；chat 省略 null 字段 #64177）、ch05（KV layout 跨 PP 合并 #61294；offload/connector 一批；DSA decode indexer 计入 profile #62860）、ch07（DBO+spec/DCP #61464 与正文冲突；idle sampler staged writes #62748；lm_head LoRA 行对齐 #59341；routed experts capturer 提前 #63237）、ch08（同 #61464/#62748；JIT warmup #63819）、ch09（同 #61464；layout #61294）。与 v0.32.0 正文描述冲突、升级时须改写正文的是 **#61464**（DBO 不再禁止 speculative decoding / DCP）。

## 影响表

| SHA / PR | 领域 | main 上的变化 | 本书位置 | 处理 |
|---|---|---|---|---|
| [458334f00f](https://github.com/vllm-project/vllm/commit/458334f00f7cbff7c8a1feae449656aa7343c476) / [#63819](https://github.com/vllm-project/vllm/pull/63819) | spec decode / MRV2 | `OnlineAcceptanceEstimator.warmup()` 在 JIT warmup 末尾编译 `_refit` 并清零统计（`acceptance_estimator.py` L440–L460 @ `36d4ef654`）；`warmup_kernels` 调用之（`warmup.py` L255–L259）；DFlash `prepare_dflash_inputs` 的 Triton `BLOCK_SIZE` 固定为 128（`dflash/speculator.py` L891）；`_accumulate_block_r` 按 `max_num_reqs` 预计算（L350） | ch08 warmup / JIT；书中未展开 acceptance estimator | 注记 |
| [06e172b5cf](https://github.com/vllm-project/vllm/commit/06e172b5cf8845c80c40a6eaaecbcbde3e38e8df) / [#59341](https://github.com/vllm-project/vllm/pull/59341) | spec decode / LoRA | `LoraState` 的 `prompt_lora_mapping` 改为 `lora_ids.repeat(num_logits)`（`lora_utils.py` L99–L104 @ `36d4ef654`），与 spec decode 下每请求 `1+num_drafts` 行 logits 对齐；`model_runner` 传入 `np.diff(cu_num_logits_np)`（L1913） | ch07 LoRA / sampling | 注记 |
| [5118d990fb](https://github.com/vllm-project/vllm/commit/5118d990fb9d28768b4ec6c7328322f96186f915) / [#62796](https://github.com/vllm-project/vllm/pull/62796) | spec decode | adaptive draft 的 prefill graph padding 与 DP 协商对齐（`dp_utils.py` + target-dependent / 通用 speculator） | 书中未展开 adaptive draft padding | 仅清单 |
| [7b87062308](https://github.com/vllm-project/vllm/commit/7b87062308855434fa0747852ccebbe950ee9e4f) / [#61699](https://github.com/vllm-project/vllm/pull/61699) | spec decode | standalone draft 经 `get_draft_load_config` 传入 `load_config`（`draft_model.py` L112 @ `36d4ef654`）；extract_hidden_states / standalone_ar 同步 | 书中未展开 standalone load | 仅清单 |
| [6ba5dbeeb0](https://github.com/vllm-project/vllm/commit/6ba5dbeeb0c11f56798c71b8855b656e6cf07b2f) / [#61270](https://github.com/vllm-project/vllm/pull/61270) | spec decode / Transformers | 新增 `transformers/mtp.py`；`SpeculativeConfig` 支持 Transformers 后端 MTP draft | 书中未展开 Transformers MTP | 仅清单 |
| [cdd2750cd5](https://github.com/vllm-project/vllm/commit/cdd2750cd5e7d90849282a16bc46c21bbc58bcbd) / [#61464](https://github.com/vllm-project/vllm/pull/61464) | MRV2 / DBO / spec / DCP | `_get_dbo_unsupported_features` **移除** “dual batch overlap with speculative decoding” 与 decode CP 禁令，仅保留 prefill CP（`vllm.py` L3247–L3270 @ `36d4ef654`）；新增 `_raise_dbo_thresholds_for_spec_decode`（L3225–L3245）按 `num_ubatches * uniform_decode_query_len` 抬高阈值；`ubatch_utils` 支持按请求边界切分多 token uniform decode 与 DCP local seq lens | ch07 §MRV2 fallback；ch08 DBO；ch09 DCP | **注记（与正文冲突）** |
| [d7f68d4c7d](https://github.com/vllm-project/vllm/commit/d7f68d4c7d31004e257b7b2ff142b4edd83e0d8c) / [#62748](https://github.com/vllm-project/vllm/pull/62748) | MRV2 / sampler | `sampler.apply_staged_writes()` 仅在本步有 `scheduled_new_reqs` 时调用（`model_runner.py` L1298–L1302 @ `36d4ef654`）；`LogitsProcessor.apply_staged_writes` 文档同步（`interface.py` L136） | ch07/ch08 staged writes | 注记 |
| [88d3995ea1](https://github.com/vllm-project/vllm/commit/88d3995ea18de47ba5c306789aa615474ad74592) / [#64186](https://github.com/vllm-project/vllm/pull/64186) | Core / sleep | 新增 opt-in `CUDACheckpointExecutor` + `CudaProcessCheckpoint`（多 worker CUDA checkpoint sleep） | ch03 sleep；书中未写该 executor | 仅清单 |
| [edad8b03c5](https://github.com/vllm-project/vllm/commit/edad8b03c5285a1c12e7e4f83c1af709b319a01b) / [#61294](https://github.com/vllm-project/vllm/pull/61294) | KV cache layout / PP | `resolve_kv_cache_layout` / `_merge_supported_layouts`：跨 PP stage 取布局交集再按多数优先，不再要求各 worker 列表完全一致（`vllm/v1/attention/backends/utils.py` L243–L337 @ `36d4ef654`） | ch05 layout 解析 | 注记 |
| [65faf4efa0](https://github.com/vllm-project/vllm/commit/65faf4efa0281fc2e04096a28cd35fb8dc7d3218) / [#63237](https://github.com/vllm-project/vllm/pull/63237) | Core / profiling | routed experts capturer（aux output connector）在 `profile_run` 中、KV cache profiling 之前初始化；`initialize_from_config` 仅 `initialize_output_storage`；`PLAN_SCHEMA_VERSION=2` | ch07 profiling 顺序 | 注记 |
| [317081fb6d](https://github.com/vllm-project/vllm/commit/317081fb6dc8b125a3134711a7750707bfd55371) / [#62860](https://github.com/vllm-project/vllm/pull/62860) | DSA / memory | decode indexer logits 计入 memory profile workspace（`sparse_attn_indexer.py` L496–L541 @ `36d4ef654`） | ch05/ch07 显存 profile | 注记 |
| [0beab6c251](https://github.com/vllm-project/vllm/commit/0beab6c251e8046c0c6cf2fef114cd23b58cda74) / [#63779](https://github.com/vllm-project/vllm/pull/63779) | KV Offload | tier promotion 可失败并 `defer_write_release`，延迟归还 chunk（`tiering/manager.py` L170–L214、L556–L638 @ `36d4ef654`） | ch05 §5.15 offload | 注记（汇总） |
| [77f58e5304](https://github.com/vllm-project/vllm/commit/77f58e530450416d217e4816727a35910bf3d9bf) / [#64229](https://github.com/vllm-project/vllm/pull/64229) | KV Offload | bounded-replay SWA 分配计入 lazy walk target（`simple_kv_offload/manager.py` L349 一带） | ch05 offload / SWA | 注记（汇总） |
| [f4fd1ab68e](https://github.com/vllm-project/vllm/commit/f4fd1ab68e2ba3a76bf6b903662eb6a8ee8202ab) / [#63207](https://github.com/vllm-project/vllm/pull/63207) | KV Offload / P2P | DEALER monitor 在关闭 reader 前先 stop | ch05 P2P | 注记（汇总） |
| [a1cfd2c1b5](https://github.com/vllm-project/vllm/commit/a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df) / [#64250](https://github.com/vllm-project/vllm/pull/64250) | KV Offload / P2P | session teardown 时保留未确认的 source WRITE | ch05 P2P | 注记（汇总） |
| [066d979934](https://github.com/vllm-project/vllm/commit/066d9799347c88c14e91baa9d849e9595ff8a3d3) / [#64230](https://github.com/vllm-project/vllm/pull/64230) | KV Connector / NIXL | `extra_config.transfer_mode` 可选 `pull`/`push`，push 注册 `NixlPushConnector`（`factory.py` L127–L129 @ `36d4ef654`） | ch05/ch09 connector | 仅清单 |
| [f79a3a7d48](https://github.com/vllm-project/vllm/commit/f79a3a7d48dad51bc12048433c1f04fc34d5942b) / [#63513](https://github.com/vllm-project/vllm/pull/63513) | KV Connector / Mooncake | store 跳过 padding KV blocks | 书中未展开 Mooncake store | 仅清单 |
| [6cfb375fd6](https://github.com/vllm-project/vllm/commit/6cfb375fd6a2c7aee76b086c19911646711834a5) / [#61793](https://github.com/vllm-project/vllm/pull/61793) | UMBP / MORI | 新增 MORI embedded DRAM runtime（`umbp/runtime/embedded.py`）并注册 `MoRIIOConnector` | 书中未展开 UMBP | 仅清单 |
| [af4db2a675](https://github.com/vllm-project/vllm/commit/af4db2a675f79ed20a8888ee0ec0c71ad581ad25) / [#61457](https://github.com/vllm-project/vllm/pull/61457) | 量化 / MoE | per-layer fused shared expert 选择（`fused_moe` + 多模型） | 书中未展开 fused shared expert | 仅清单 |
| [e10ef091ba](https://github.com/vllm-project/vllm/commit/e10ef091baf01b3ba17b93e49e0a94228c36cbfb) / [#62407](https://github.com/vllm-project/vllm/pull/62407) | 量化 / 权重 | `process_weights_after_loading` 后保留已删除/重命名参数注册（mixed_precision / scaled_mm / MoE oracle） | 书中未展开 parameter 注册 | 仅清单 |
| [2b678ac739](https://github.com/vllm-project/vllm/commit/2b678ac739166bd0356883048673bbd1c34a3b0e) / [#62557](https://github.com/vllm-project/vllm/pull/62557) | RL / 权重 | NCCL weight transfer 支持 per-update source ranks | 书中未讲 RL weight transfer | 仅清单 |
| [c91ab83dce](https://github.com/vllm-project/vllm/commit/c91ab83dced6f9a5b5227c6f83595abbf4868d6e) / [#64287](https://github.com/vllm-project/vllm/pull/64287) | 模型 | 删除 `keye.py` / `keye_vl1_5.py`；registry 将 Keye* 移入 `_PREVIOUSLY_SUPPORTED_MODELS`（L844；Keye* 于 L898–L899 注明自 0.33.0 移除，`registry.py` @ `36d4ef654`） | 书中未列 Keye-VL | 仅清单 |
| [ee9ba37199](https://github.com/vllm-project/vllm/commit/ee9ba37199338cf31306a2ec27ec9afa41efc09e) / [#64282](https://github.com/vllm-project/vllm/pull/64282) | 模型 | 新增 Nemotron-3.5-Transcribe（含 MTP 注册，`registry.py` L549–L551、L727–L729） | 书中未列该模型 | 仅清单 |
| [9fc019a749](https://github.com/vllm-project/vllm/commit/9fc019a749cbf3235cc06aa5357b17fd37cb648a) / [#64097](https://github.com/vllm-project/vllm/pull/64097) | Frontend | `run_batch` 支持 `/v1/completions`（`run_batch.py` L182–L197、L828–L829 @ `36d4ef654`） | ch02 在线入口 | 注记 |
| [f60edccd00](https://github.com/vllm-project/vllm/commit/f60edccd0079428503f9e9f66e3b651aa4da73db) / [#64177](https://github.com/vllm-project/vllm/pull/64177) | Frontend | chat completions 响应省略 null 的 `annotations` 与 `usage.prompt_tokens_details`（`exclude_none`，`chat_completion/protocol.py` L1269 一带） | ch02 chat API | 注记 |
| Rust Frontend 一批（#63959–#63964、#59694） | Rust Frontend | 与 Python 对齐：sleep 指标、kv_fetch_by_stage、LoRA 404、reasoning tokens、list chat templates、`max_tokens` 上限、bench prompt_tokens | 书中未讲 Rust 前端 | 仅清单（合并） |
| HiSparse 修复一批（#64238、#63827、#61659、#62739） | HiSparse | compact KV 测试、defer async load、staged prefill pad、resident row by request slot | ch05 HiSparse 未细化 | 仅清单（合并） |
| [b766f15983](https://github.com/vllm-project/vllm/commit/b766f15983) / [#64020](https://github.com/vllm-project/vllm/pull/64020) | Frontend | Anthropic inline-system probe 启动崩溃修复 | 书中未讲 Anthropic probe | 仅清单 |

## 不受影响（本区间相对书中锚点）

- ch01、ch04、ch06、ch10：本区间无改写其 v0.32.0 正文锚点符号的提交（ch10 实验合同无需新注记）。
- ch03：CUDA checkpoint sleep（#64186）为新 opt-in 路径，不改现有 sleep/wake 正文描述。
- 纯 CI / ROCm / XPU / MyPy / Build / Doc recipe / Security（ReDoS、FunAudio 时长）/ Bench 默认 / 具体 Attention·Pooling·LoRA EP 边角：不影响章节正文，一句带过。

## 升级清单（下一个 tag 发版后）

- [ ] 确认新 tag 是否包含上表各提交。
- [ ] ch07：按 #61464 改写 DBO 组合校验——speculation 与 DCP 不再在 `_get_dbo_unsupported_features` 中；写明阈值抬高与 ubatch 按请求切分多 token decode。
- [ ] ch08：同步 DBO+spec；`apply_staged_writes` 仅在有新请求时刷 sampler；JIT warmup 调用 `acceptance_estimator.warmup()`。
- [ ] ch05：按 #61294 改写跨 PP layout 解析（交集+多数优先）；并入 offload #63779/#64229/#63207/#64250；DSA #62860。
- [ ] ch07：#59341 lm_head LoRA 按 logits 行展开；#63237 capturer 在 profile 前初始化；#62748。
- [ ] ch02：run-batch `/v1/completions`；chat 省略 null 字段。
- [ ] 模型清单（若有）：Keye-VL 移除（自 0.33.0）、Nemotron-3.5-Transcribe 新增。
- [ ] 更新 chapter-source-map.json 的 source_commit，跑 check_book.py 并新增阅读基线 004。
