# vLLM main 跟踪记录（2026-10-29～2026-10-30，未发版）

- 当前阅读基线：`v0.32.0` / `5df63a7203cdb9328d824a27b682e77c7b9928be`（基线 003），正文不变。上游仍无新 GitHub Release（最新仍是 v0.32.0，2026-10-18）。区间附近只有 git rc tag：`v0.32.1rc0`（2365177dd，2026-10-18，在 main 上、早于本区间起点 36d4ef654）、`v0.33.0rc0`（4135590ff，2026-10-26，在 main 上、早于区间起点）、`v0.33.0rc1`（cbafc4690，2026-10-28，不在 main 历史上，位于发布分支）；三者都没有 GitHub Release 条目（连 pre-release 也没有），阅读基线不变。
- 跟踪区间：`vllm-project/vllm` `main` `36d4ef654a83e227216b0c0b55cf4041d96b1675`..`2eae56c4ba0cf12087fec035514eedede83c257d`（[#64610](https://github.com/vllm-project/vllm/pull/64610)，2026-10-30 11:27 UTC+8），共 109 个提交。接续 [2026-10-29 记录](main-tracking-2026-10-29.md)。
- 方法：在只读 partial clone 中逐条 `git show` 阅读提交的非测试源码 diff；只记录读过 diff 的条目。没有运行上游代码、GPU 或模型。109 个提交全部覆盖：影响表 24 行（重点领域）+ 分组行（offload/connector/模型/frontend）+「一句带过」分组 + 文末「其余 53 个提交（逐条一句）」。行号均按 tip `2eae56c4b` 核对。
- 结论：v0.32.0 正文对 v0.32.0 本身仍成立。建议加“main 上的变化（未发版）”注记：ch02（Responses `thinking_token_budget` #63896）、ch05（offload/connector 一批；DSA/DSv4.1 profile #63047）、ch06（`AttentionBackend.supports_head_size_v` #64064）、ch07（PP 下 structured output 逐请求 draft 保留 #63219；PP draft/sampled broadcast #64458）、ch08（同 #63219；Qwen3.5 弃用 `torch.compile` / 默认 breakable CG #51012；#64279 撤销 #42129 allow-list，注意 v0.32.0 基线自身带该 allow-list）、ch09（同 #64458；#64356 替换 #63554 的 owner/row 规则）。本区间**没有**与 v0.32.0 正文描述直接冲突、升级时必须改写正文的提交（#63219 / #64458 / #51012 是机制补充或模型默认路径切换，不是推翻现有主路径叙述；#64279 修的是基线已有的 allow-list 问题）。

## 影响表

| SHA / PR | 领域 | main 上的变化 | 本书位置 | 处理 |
|---|---|---|---|---|
| [322f90a2a](https://github.com/vllm-project/vllm/commit/322f90a2a356a83276bc07236e0e620a4e4fc93a) / [#63219](https://github.com/vllm-project/vllm/pull/63219) | spec decode / PP / structured output | `GrammarOutput` **删除** `num_acceptable_drafts`（`output.py` L341–L346 @ `2eae56c4b`）；`Scheduler` 只返回 bitmask（`scheduler.py` L2245–L2250）。`DraftTokensHandler` 改为按请求保留 draft：`_draft_tokens_by_req` + `begin_step` / `discard` / `get_draft_tokens`（`spec_decode/utils.py` L51–L131）；`execute_model` 开头 `begin_step`（`model_runner.py` L1794–L1795），`_remove_request` 时 `discard`（L1198–L1199）。删除 `grammar_invalid_drafts` 与 `RejectionSampler` 的 `invalid_drafts` / `verify_draft_sampled` 旁路。PP 下 structured output 不再依赖“当前 batch 整表 draft”，而是按 req_id 取上一步保留的 draft | ch07 §7.19 grammar / §7.20 PP sample；ch08 §8.3.4 deferred grammar | 注记 |
| [d8fdf4f61](https://github.com/vllm-project/vllm/commit/d8fdf4f61bb606e2eb216230cd821096b2a64d3e) / [#64458](https://github.com/vllm-project/vllm/pull/64458) | MRV2 / PP / spec | `PPHandler._sample_len`：有 draft 时广播宽为 `max_sample_len`，否则 1（`pp_utils.py` L104–L106）。`get_prev_sampled_outputs` 不再原地写 `draft_tokens_to_update`，改为把 `slot.draft_tokens` 交给 `postprocess_sampled`（L131–L163）。`broadcast_drafts` 只广播本步 `draft_tokens` 张量（L165–L174）；`receive(..., recv_drafts=)` 与 last rank 是否 `broadcast_drafts` 对齐（L176–L231）。`post_update` kernel 把 draft 写入 `req_states.draft_tokens`（`input_batch.py`）。仅当 `num_spec_tokens > 0` 才 broadcast drafts（`model_runner.py`） | ch07 §7.15/§7.20；ch09 §9.6 PP | 注记 |
| [66875d312](https://github.com/vllm-project/vllm/commit/66875d3127534183b5ea723964fc7ec388e89722) / [#64356](https://github.com/vllm-project/vllm/pull/64356) | spec decode / TP shard | 新增 `DraftLogitsCache`（`vllm/v1/worker/gpu/spec_decode/draft_logits.py` L80–L185）。分片 cache：每步 draft 的 `assign` 把 in-batch 位置 i 的请求交给 rank `i % tp`、行 `i // tp`（L84–L90、L139–L156；`shard_owners`/`shard_rows` 在 `vllm/v1/worker/gpu/sample/batch_shard.py` L67–L78 现按 batch position，不再按 `req_state_idx`），owner 记入每请求 slot 的 `shard_mapping`；verification 用记录的 owner（`owners_of`，L158–L173）；行每步复用，`stale_drafts` 标记行被改派的 draft，verification 拒绝它们（L92–L95、L175–）。复制 cache（或无 draft logits）：`BatchSharder` 按 in-batch 位置 `shard_owners(num_reqs, tp)` 轮转分配（`batch_shard.py` L238–L246）；此前是 `idx_mapping_np % tp_size`（按 req_state slot）。从未 draft 的请求仍按 slot 定 owner（L118–L122） | ch09 §9.6.6 batch-sharded sampling 注记（接 #63554 / #64151） | 注记 |
| [0d6f92ae7](https://github.com/vllm-project/vllm/commit/0d6f92ae7c83f1d338d1b6a742ebcd79ab0fc2bb) / [#64137](https://github.com/vllm-project/vllm/pull/64137) | spec decode | MRV2 logits-stats Triton kernel 不再按 `num_logits` 特化 | 书中未展开 rejection stats kernel | 仅清单 |
| [cebe10972](https://github.com/vllm-project/vllm/commit/cebe1097232eaaba46aeae230dfe0887c88f634f) / [#60229](https://github.com/vllm-project/vllm/pull/60229) | spec decode | GLM-5.3-Flash / GDN / sparse indexer 支持 adaptive verification 的 decode seq lens | 书中未展开 adaptive verification metadata | 仅清单 |
| [d295942d0](https://github.com/vllm-project/vllm/commit/d295942d0d9dfd25e1daf09cb1156c7a5c43c9d6) / [#63275](https://github.com/vllm-project/vllm/pull/63275) | spec decode / trace replay | `trace_replay_state.apply_trace_to_verified` 接入 rejection sampler；`sampling_params` 去掉旧限制 | 书中未讲 trace replay + spec | 仅清单 |
| [b7f0dda50](https://github.com/vllm-project/vllm/commit/b7f0dda503ef69a9784b1fb8a6068a32ebc80ae9) / [#62990](https://github.com/vllm-project/vllm/pull/62990) | ROCm / spec / DCP | `ROCM_AITER_MLA` 允许 DSpark adaptive verification 与 DCP 组合 | 书中未展开 ROCm MLA + DSpark | 仅清单 |
| [e49ea84e9](https://github.com/vllm-project/vllm/commit/e49ea84e929e3d48a2bc500454dfcd4377f9a722) / [#63885](https://github.com/vllm-project/vllm/pull/63885) | ROCm / DSv4.1 / SWA | 新增 `swa_decode_meta_enabled()`：`--monokernel` 且 CDNA3/4（gfx942/gfx950）时，decode SWA metadata 单次 launch 构建（`deepseek_v41/amd/ops/__init__.py` L49–L58；`swa_decode_meta.py` `build`）。其余 gfx942 helper 仍只认 `enabled()`=gfx942 | 书中未展开 DSv4.1 ROCm SWA metadata | 仅清单 |
| [d6fbd9478](https://github.com/vllm-project/vllm/commit/d6fbd947824959eabc6165751f8b9cd4aa83f546) / [#51012](https://github.com/vllm-project/vllm/pull/51012) | 模型 / compile | CUDA 上 Qwen3.5 六个 architecture 并入 `default_breakable_cudagraph_architectures()`（`vllm/config/vllm.py` L76–L84、L140–L143 @ `2eae56c4b`），走 eager + breakable CUDA Graph；registry 改指向 `vllm.models.qwen3_5`（CUDA 用 `nvidia/model.py`，非 CUDA 仍用旧 `torch.compile` 实现）。不是删除 `torch.compile` 机制本身 | ch08 breakable CG / `@support_torch_compile` 默认路径 | 注记 |
| [a6c3635e3](https://github.com/vllm-project/vllm/commit/a6c3635e3660f6343c14e2158c7ab44c62be88b9) / [#64064](https://github.com/vllm-project/vllm/pull/64064) | Attention backend / ROCm | `AttentionBackend.supports_head_size_v` 默认要求 V=K（`backend.py` L109–L112）；`RocmAiterUnifiedAttentionBackend` 允许 `head_size_v <= head_size`（`rocm_aiter_unified_attn.py` L90–L94）。MiMo-V2.6 在用户指定 backend 能服务 DiffKV 几何时不再强制 `*_DIFFKV`。`csrc/cache_kernels.cu` 新增 `reshape_and_cache_flash_kernel_asym`（L428），仅当 `v_head_size != head_size` 时由 `reshape_and_cache_flash` 使用（dispatch L993–L1014）；K/V 须共享 head stride / page stride / block geometry，对称 kernel 路径不变 | ch06 §6.14.1 backend 能力 | 注记 |
| [d1ad431de](https://github.com/vllm-project/vllm/commit/d1ad431de60f995585e75d66bb5bea55f313ca96) / [#64279](https://github.com/vllm-project/vllm/pull/64279) | compile | **撤销** #42129（b4b4aaa70）加入的 `_VllmFallbackAllowList`：该代理把所有 `vllm::*` / `vllm_aiter::*` op 当作 Inductor `FALLBACK_ALLOW_LIST` 成员，allow-list 内 op 的 fallback 不带 PyTorch 对 custom op 的默认 layout 约束，于是 `vllm::unified_kv_cache_update` 收到 head-major 的 key（stride `(128, 128*S, 1)`，eager 为 `(1024, 128, 1)`）；`reshape_and_cache_flash` 只读 `key.stride(0)`（`csrc/cache_kernels.cu` L977）、假设 head 紧排，KV cache 被写乱，输出退化。触发条件：Transformers backend（`model_impl=transformers`）+ compile mode 1/2，或 mode 3 且 `splitting_ops: []`（整图编译把 attention 与 KV 写入编进同一个 Inductor graph）；默认 mode 3 在 attention/KV-cache op 处切图，不受影响。v0.32.0 本身仍带该 allow-list（`env_override.py` L855–L917 @ v0.32.0），即阅读基线也受影响。保留下来的 `OperatorIssue.operator_str` 补丁只是 #42129 原本的防挂起目的：torch &lt; 2.13 上对 vllm op 只打印 target（`env_override.py` L833–L888，`is_torch_equal_or_newer("2.13.0.dev")` 时直接返回），不是本 commit 的主体。回归测试 `tests/models/transformers/test_backend.py::test_whole_graph_compile`（Qwen3-0.6B，mode 2 vs eager） | ch08 §8.8.1 splitting ops | 注记 |
| [f1266682b](https://github.com/vllm-project/vllm/commit/f1266682b26d5242e9ba00d2fb15a5c93a26a86a) / [#64423](https://github.com/vllm-project/vllm/pull/64423) | ROCm / DBO | `UBatchRunner` capture 前在微批线程里 `torch.mm` 预创建 hipBLASLt handle（`ubatch_utils.py` L509–L519） | ch08 DBO capture；ROCm 特化 | 仅清单 |
| [5c1939140](https://github.com/vllm-project/vllm/commit/5c19391400c58ae10feca2483f55c547df211715) / [#63896](https://github.com/vllm-project/vllm/pull/63896) | Frontend | `ResponsesRequest.thinking_token_budget` 传入 `SamplingParams`（`responses/protocol.py` L452、L650 @ `2eae56c4b`），与 chat completions 对齐。Rust frontend 仍不服务 `/v1/responses` | ch02 在线协议 | 注记 |
| [7ab756940](https://github.com/vllm-project/vllm/commit/7ab756940a89a7464ebd6ab26c7c6a1f3b7d08da) / [#63047](https://github.com/vllm-project/vllm/pull/63047) | DSA / DSv4.1 / memory | `sparse_attn_indexer` profile dummy 把 prefill FP32 copy 计入 workspace（`sparse_attn_indexer.py` L519–L543 @ `2eae56c4b`），不再只按 decode logits 估峰值；DSv4.1 Engram 在 profile 跑 dummy hashes。接续 10-29 的 #62860 | ch05/ch07 显存 profile | 注记 |
| [1aea05364](https://github.com/vllm-project/vllm/commit/1aea05364f4e24bfb249077880c6de6f5a2c21fd) / [#60477](https://github.com/vllm-project/vllm/pull/60477) | DSA / PP | 非末 PP stage 发送前做 `hidden_states = residual + tensor_model_parallel_all_reduce(hidden_states)`，`IntermediateTensors` 只带 `hidden_states`（不再带 `residual`），后续 stage 把收到的值当作已 TP-reduce 的残差流（`residual = None`）。DeepseekV32 nvidia（`vllm/models/deepseek_v32/nvidia/model.py` L232、L300）与 amd（`amd/model.py` L190、L233）；Dots3Note 只改 `make_empty_intermediate_tensors` key（`dots3_note/nvidia/model.py` L615）。nvidia 路径同时删了 "SP is not supported with PP" 断言 | 书中未把 residual key 写成通用契约 | 仅清单 |
| Offload 一批（#64400、#59099、#63618、#62930、#63566、#63788、#64251、#59322、#63937、#64501、#64345、#62444、#64256、#64224、#52838） | KV Offload / P2P | in-flight partial-tail 去重；`reset_cache` 可重入；idle P2P-only 可 park；跨节点拒绝 secondary tier；sparse 切中前缀时 pin junction；OBJ telemetry 缺失不丢完成事件；WRITE stall 判 peer 故障；避免重复 hash-prefix 扫描；出错请求不 offload 半块；handshake 未完成则本地 prefill；SimpleCPUOffload 不去零填充 / GPU-hit 刷新 LRU；P2P 超时立即失败并延迟释放 slot；KVCR v0.2.0；OffloadingConnector 传输故障降级缓存而不是打垮 engine | ch05 §5.15/§5.17 offload | 注记（汇总） |
| Connector 一批（#64582、#64193、#57662、#62905、#59934） | KV Connector | NIXL：block-outermost Mamba stride；按 `(base_addr, block_len)` 去重 region；`ucx_backend_params`。Mooncake：decoder 续 prefill lease；`MooncakeStoreConnector` 支持 sleep | ch05/ch09 connector；书中未展开 NIXL/Mooncake 细节 | 仅清单（合并） |
| [28dd46d79](https://github.com/vllm-project/vllm/commit/28dd46d791981ad062be76f6ed4f0b3e2938164a) / [#60103](https://github.com/vllm-project/vllm/pull/60103) | HiSparse | 融合 per-layer slot invalidation，缩短 all-resident resolver | ch05 HiSparse 未细化 | 仅清单 |
| [f5aa164b5](https://github.com/vllm-project/vllm/commit/f5aa164b514819fea7dffa245457f04965cb9309) / [#63489](https://github.com/vllm-project/vllm/pull/63489) | LoRA | `LoRAModelManager.unsupported_lora_modules`：同后缀但未 wrap 的层（如 BERT `pooler.dense`）上的 adapter 直接拒绝（`model_manager.py` L485–L505） | ch07 LoRA；书中未讲加载拒绝规则 | 仅清单 |
| [a8a0520e4](https://github.com/vllm-project/vllm/commit/a8a0520e4336c1b212f980e4335c3a7e718949a4) / [#57990](https://github.com/vllm-project/vllm/pull/57990) | Frontend / derender | `/v1/completions/derender` 保留 engine 已算的 `prompt_logprobs`；`OnlineDerenderer.derender_chat` 返回 `(choices, prompt_logprobs)` | 书中未讲 scale-out derender | 仅清单 |
| [d8e6fbe46](https://github.com/vllm-project/vllm/commit/d8e6fbe461f54ec17274fc49000e94d332347e6a) / [#64367](https://github.com/vllm-project/vllm/pull/64367) | Frontend / renderer | `HfRenderer` 把 prompt-embeds sentinel 换成 0 再交 engine（`hf.py` L1482、L1504），避免 OOV id 进模型 | ch02 Renderer；书中未讲 prompt embeds sentinel | 仅清单 |
| [4e0ad5720](https://github.com/vllm-project/vllm/commit/4e0ad572036166a8bdf1782f4a07b4e33016646b) / [#62359](https://github.com/vllm-project/vllm/pull/62359) | DSv4.1 / DP | Engram host table 经 host-NUMA cuMem 在 DP ranks 间共享 | 书中未展开 Engram | 仅清单 |
| [98fc6da9d](https://github.com/vllm-project/vllm/commit/98fc6da9d7355d2cc85141980439fded9a5fb61c) / [#59131](https://github.com/vllm-project/vllm/pull/59131) | DSv4.1 | prefill 按 KV-source group 只 gather 一次 compressed KV | 书中未展开 DSv4.1 gather | 仅清单 |
| [b2f46e785](https://github.com/vllm-project/vllm/commit/b2f46e785fe1584ecc654e5ef1bdc3fccc2c597a) / [#62009](https://github.com/vllm-project/vllm/pull/62009) | EPLB / DSv4 | 新增 `symm_mem` weight-migration backend，融合 MegaMoE replica mapping | ch09 EPLB 未写该 backend | 仅清单 |
| [61a6f8138](https://github.com/vllm-project/vllm/commit/61a6f8138fd81f89467af762dc1b74674f544300) / [#61777](https://github.com/vllm-project/vllm/pull/61777) | Bench | Rust `vllm-bench` profiling 聚合 DP running requests | ch10 Rust bench 未展开 | 仅清单 |
| [03098917a](https://github.com/vllm-project/vllm/commit/03098917ad67740d15a97ae0ce32ecc05fbb25a0) / [#57022](https://github.com/vllm-project/vllm/pull/57022) | RL / 权重 | NCCL weight transfer 失败改为显式报错，不再 no-op | 书中未讲 RL weight transfer | 仅清单 |
| 模型两个（#64111、#63585） | 模型 | MiniMax-M3.1-Flash（`registry.py` MiniMaxM3Sparse* / MTP）；Ming3-ASR（L517–L519 `MingV3ASRForConditionalGeneration`） | 书中未列这些模型 | 仅清单（合并） |
| Frontend / Parser / Security 一批（#64615、#64535、#58780、#64503、#63594、#64169） | Frontend | GLM-4.7 tool 参数不 pin / 闭包只 parse 一次；DeepSeek V4/V3.2 字符串 tool 增量流式；structured decisions 强制 request template trust；Anthropic 非 jinja inline-system probe；MistralParser 走 `get_grammar_for_chat_request` | 书中未展开这些 parser | 仅清单（合并） |

## 不受影响的章节

- **ch01**：本区间没有改写自回归、prefill/decode、KV 容量直觉或 sampling 主锚点（`SamplingParams` / sampler 接口）的提交。
- **ch03**：没有改 EngineCore 初始化、IPC、busy loop、pause/sleep/wake 契约；#59934 只让 MooncakeStoreConnector 跟随已有 sleep，不改 ch03 正文描述的 sleep/wake。
- **ch04**：#63219 只让 `Scheduler` 不再填充 `GrammarOutput.num_acceptable_drafts`，`schedule()` 的 RUNNING/WAITING/token budget/preemption 主循环未改；书中也未引用该字段。
- **ch10**：#61777 只改 Rust bench 的 DP running-request 聚合；实验合同、指标定义与 warmup 叙述无需新注记。

## 一句带过

纯 CI / ROCm 镜像与 AITER nightly（#64610）、ROCm kernel/quant/FlyDSL/Qwen4Exp/Kimi-K3 测试、XPU mrope/UT、recipe 文档（Mistral Large 4、DeepSeek-V4-Pro MI355X、k8s-aibom、iluvatar）、Transformers 5.21.1、mypy、自动分片、Rust frontend bench 边角（#64557、#64390）、官方 V1 architecture overview 补 Scheduler 节（#38741）：不影响章节正文。上一句按主题归并；下面把影响表未点名的其余 53 个提交按 PR 号各写一句。

## 其余 53 个提交（逐条一句）

- [#53011](https://github.com/vllm-project/vllm/pull/53011)（`604b96164`）：删除 Kimi K2.5 vision helper `get_rope_shape` 上的 eager `@torch.compile` 装饰与 warmup wrapper；不影响正文。
- [#57924](https://github.com/vllm-project/vllm/pull/57924)（`5a4f307d9`）：Nemotron VL 图像归一化可移到 device（`FusedMMInputNorm` + processor `do_cpu_normalize`）；不影响正文。
- [#58468](https://github.com/vllm-project/vllm/pull/58468)（`d83a7df60`）：RADIO vision encoder 按当前 Transformers 重写（`radio.py` / config），Nemotron VL/Parse 权重映射跟着改；不影响正文。
- [#59938](https://github.com/vllm-project/vllm/pull/59938)（`34091a07c`）：XPU mrope 增加 `_xpu_C.multimodal_rotary_embedding` 路径，并把文本 RoPE 抽到 `_forward_text_rope`；XPU 特化，不影响正文。
- [#60163](https://github.com/vllm-project/vllm/pull/60163)（`5cf5c8fa6`）：Humming FP8 标 `processed_weight_dims`、视为 batch-invariant，并禁止当 PTPC 的 W8A16 内核；量化边角，不影响正文。
- [#60899](https://github.com/vllm-project/vllm/pull/60899)（`dc45a5f55`）：新增 k8s-aibom 部署集成文档；不影响正文。
- [#61107](https://github.com/vllm-project/vllm/pull/61107)（`18450e5ef`）：Qwen2 MoE shared MLP 的 `act_fn` 改走 `maybe_fused_act_quant`；模型边角，不影响正文。
- [#61529](https://github.com/vllm-project/vllm/pull/61529)（`45fd60d4d`）：AMD Qwen4Exp QSA 路径支持 fp8 KV / indexer cache（`indexer_kv_dtype`、kernel `IS_FP8`）；ROCm 模型特化，不影响正文。
- [#61939](https://github.com/vllm-project/vllm/pull/61939)（`bba4d9afe`）：ROCm CI/release 镜像层级合并，Dockerfile / bake 流水线统一；不影响正文。
- [#62481](https://github.com/vllm-project/vllm/pull/62481)（`0124e16a6`）：`jit_monitor` 增加 FlyDSL `MlirCompiler.compile` hook，warmup 后监视 JIT；与 ch08 JIT warmup 弱相关，不影响正文。
- [#62500](https://github.com/vllm-project/vllm/pull/62500)（`5082c4585`）：`custom_op.md` 补 vllm-iluvatar；不影响正文。
- [#62976](https://github.com/vllm-project/vllm/pull/62976)（`6c71831f2`）：AITER_MOE_SORT_BACKEND MXFP4 DSv4 测试覆盖；只改测试，不影响正文。
- [#62996](https://github.com/vllm-project/vllm/pull/62996)（`e3f03f4f8`）：取消 AITER MXFP4 padding-matrix / token-padding 测试 skip；只改测试，不影响正文。
- [#63053](https://github.com/vllm-project/vllm/pull/63053)（`a0e31b735`）：AITER custom AR 不再用于 stateless/elastic DP group；elastic EP 切图时换新 CUDA graph pool；与 ch09 elastic EP 弱相关，不影响正文。
- [#63054](https://github.com/vllm-project/vllm/pull/63054)（`f81a414fa`）：Triton W4A16 GEMM 的行偏移改为 int64，避免大 M 溢出；ROCm kernel 边角，不影响正文。
- [#63179](https://github.com/vllm-project/vllm/pull/63179)（`abaec97b4`）：ROCm MXFP8 block32 GEMM 把 tiny-M 量化融进 kernel；lm_head skinny-GEMM 按 `VLLM_SKINNY_GEMM_MAX_WEIGHT_BYTES` 限权重大小；ROCm 量化，不影响正文。
- [#63218](https://github.com/vllm-project/vllm/pull/63218)（`cf3f6405b`）：Nemotron Omni 三个 architecture 挂到已有的 `NemotronHNanoVLV2Config`（沿用其 `mamba_ssm_cache_dtype` 默认）；不影响正文。
- [#63286](https://github.com/vllm-project/vllm/pull/63286)（`dc9278ee2`）：SEQ_CLS/TOKEN_CLS LoRA 即使 config 省略 `modules_to_save` 也加载 `classifier`/`score`；与 ch07 LoRA 弱相关，不影响正文。
- [#63382](https://github.com/vllm-project/vllm/pull/63382)（`b42f25b00`）：NVFP4 fused silu+quant kernel 覆盖 padded scale 行/列并写零，避免未初始化 scale 出 NaN；量化 kernel，不影响正文。
- [#63384](https://github.com/vllm-project/vllm/pull/63384)（`bc57dc9d5`）：删除专用 `olmoe.py`，OLMoE 改走 Transformers backend MoE，并修 MoE LoRA；模型路径切换，不影响正文。
- [#63816](https://github.com/vllm-project/vllm/pull/63816)（`e18e2c70c`）：DSv4.1 gfx942 mono routing 对 NaN score 置 0 / 不胜出，ballot 同时限制 `cursor < ept`；ROCm 模型边角，不影响正文。
- [#63915](https://github.com/vllm-project/vllm/pull/63915)（`bcbcd02e5`）：ROCm MXFP8 1x32 `dot_scaled` 在 M≤16 时把 activation quant 融进 linear；ROCm 量化，不影响正文。
- [#64008](https://github.com/vllm-project/vllm/pull/64008)（`fb7def9ef`）：FlyDSL host launcher 复用已编译产物（`flydsl_kernels._run_compiled`，DSv4/DSv4.1/Kimi-K3 mono 调用之）；ROCm 性能，不影响正文。
- [#64013](https://github.com/vllm-project/vllm/pull/64013)（`8d3f6f7e7`）：penalties Triton kernel 给链式 `or` 加括号，兼容 Ascend；与 ch07 sampling 弱相关，不影响正文。
- [#64019](https://github.com/vllm-project/vllm/pull/64019)（`9a0f9b45d`）：`StageMissingLayer.tie_weights` 空操作，省略的 LM head 不再做 weight tying；pooling/PP 边角，不影响正文。
- [#64037](https://github.com/vllm-project/vllm/pull/64037)（`a94b768d9`）：MiniMax-M3 Triton indexer 在 SM90 上启用 split-K（`score_target_grid=1056`）；模型 kernel，不影响正文。
- [#64049](https://github.com/vllm-project/vllm/pull/64049)（`8d4c96804`）：Transformers 上界升到 `< 5.22.0`；`minicpmv4_6` 在 ≥5.21 时把 `downsample_mode` 传给 resize；不影响正文。
- [#64071](https://github.com/vllm-project/vllm/pull/64071)（`a17e79abd`）：GLM-5.3-Flash EPD 支持 metadata-only image/video 输入（`glm4_1v.py` / `glm5next` multimodal）；不影响正文。
- [#64101](https://github.com/vllm-project/vllm/pull/64101)（`f97a8d664`）：DSv4.1 AMD mono decode 支持 MI355X TP8（attention back/plan/gemv）；ROCm 模型，不影响正文。
- [#64105](https://github.com/vllm-project/vllm/pull/64105)（`3966924d5`）：MiniCPM OCR token 估算传入 `downsample_mode`；不影响正文。
- [#64113](https://github.com/vllm-project/vllm/pull/64113)（`24e80c261`）：mergify 增加 nixl auto-label 与 topic owners；不影响正文。
- [#64115](https://github.com/vllm-project/vllm/pull/64115)（`434eadcbf`）：kernels/lm_eval/models_multimodal 两步与两个 AMD mirror 纳入自动分片；不影响正文。
- [#64228](https://github.com/vllm-project/vllm/pull/64228)（`38b177540`）：XPU 复用 `test_phi3v.py::test_models_text`，去掉 XPU skip；只改测试，不影响正文。
- [#64322](https://github.com/vllm-project/vllm/pull/64322)（`d04560f56`）：DeepSeek-V4-Pro recipe 增加 MI355X AgentX source profiles；不影响正文。
- [#64342](https://github.com/vllm-project/vllm/pull/64342)（`83bae4923`）：修复多模态测试目录 mypy；`Field(default=...)`、`SupportsLoRA.lora_manager` 默认 None 等类型微调；不影响正文。
- [#64362](https://github.com/vllm-project/vllm/pull/64362)（`9957f7461`）：新增 OpenVLA chat template 并注册 fallback；不影响正文。
- [#64378](https://github.com/vllm-project/vllm/pull/64378)（`2abde627d`）：NVFP4 MoE 走 online 再量化（`requantization_source_scope`），并新增 fused shared experts（`online/shared_expert.py`）；量化边角，不影响正文。
- [#64411](https://github.com/vllm-project/vllm/pull/64411)（`28a5fba06`）：DeepGEMM 上游 tag 从 `082003075` 更新为 `5cdb14570`；不影响正文。
- [#64419](https://github.com/vllm-project/vllm/pull/64419)（`fa1cf1d43`）：GDN core 输出改为未初始化分配，FlashInfer prefill 原地写；Qwen GDN 性能，不影响正文。
- [#64425](https://github.com/vllm-project/vllm/pull/64425)（`914d61fc5`）：Qwen3.5 GDN 在 CUDA Graph capture 下用 aux stream 重叠 `in_proj_ba` 与 `in_proj_qkvz`（`overlap_ba_proj=True`）；模型性能，不影响正文。
- [#64474](https://github.com/vllm-project/vllm/pull/64474)（`ee1056b31`）：MiniMax-M3 AMD docstring 交叉引用标点修正；不影响正文。
- [#64481](https://github.com/vllm-project/vllm/pull/64481)（`1ca0dfb97`）：GLM-5.3 低延迟 BF16 GEMM 在 SM100 启用（`glm52_low_latency_gemm.py`）；不影响正文。
- [#64482](https://github.com/vllm-project/vllm/pull/64482)（`3765000a3`）：Qwen4Exp SM100 skinny GEMM 增加 TP=1 计划表；不影响正文。
- [#64506](https://github.com/vllm-project/vllm/pull/64506)（`510d15c49`）：Nemotron 3.5 Super VL 优先读原生 Transformers 的 `vision_hidden_size`；不影响正文。
- [#64525](https://github.com/vllm-project/vllm/pull/64525)（`262eefc9f`）：Muse Spark NVFP4 增加 hybrid MegaMoE/TRT-LLM 执行（`hybrid_moe.py`）；模型边角，不影响正文。
- [#64527](https://github.com/vllm-project/vllm/pull/64527)（`5cdbd0520`）：新增 Mistral-Large-4-1T-A52B recipe；不影响正文。
- [#64532](https://github.com/vllm-project/vllm/pull/64532)（`de83f5584`）：抬高 gfx942 draft lm_head 测试的 skinny-GEMM 上限；只改测试，不影响正文。
- [#64544](https://github.com/vllm-project/vllm/pull/64544)（`10579b71e`）：Mistral Large 4 recipe 更新 ROCm 镜像 tag / MI355X / TP4 replica；不影响正文。
- [#64553](https://github.com/vllm-project/vllm/pull/64553)（`36ed4327d`）：新增 Muse Spark Open CUDA routing kernel（`muse_spark_routing.cu`，经 `_moe_C` / `fused_routing`）；不影响正文。
- [#64554](https://github.com/vllm-project/vllm/pull/64554)（`b91df5a9f`）：文档构建门禁改为对 merge-base `origin/main...HEAD` 做 diff；不影响正文。
- [#64555](https://github.com/vllm-project/vllm/pull/64555)（`4b47db922`）：P2P idle peer-hit 测试等待 handshake；只改测试，不影响正文。
- [#64576](https://github.com/vllm-project/vllm/pull/64576)（`f2ba523d6`）：每个 MoE kernel 测试文件开始前清空 GPU 状态；只改测试，不影响正文。
- [#64619](https://github.com/vllm-project/vllm/pull/64619)（`daccf69eb`）：Kimi-K3 KDA checkpoint 测试 stub 补 `enable_adaptive_verification`；只改测试，不影响正文。

## 升级清单（下一个 tag 发版后）

- [ ] 确认新 tag 是否包含上表各提交。
- [ ] ch07：按 #63219 改写 grammar / spec draft 生命周期——`DraftTokensHandler` 按 req_id 保留；删除 `num_acceptable_drafts` 与 `grammar_invalid_drafts`。按 #64458 改写 PP 下 draft/sampled-token 广播（`_sample_len`、`recv_drafts`、`post_update` 写 draft）。
- [ ] ch08：§8.3.4 补上 deferred grammar 依赖“逐请求保留的 draft”，而不是当前 batch 整表；#51012 写明 CUDA 上 Qwen3.5 默认 breakable CG、非 CUDA 仍 `torch.compile`。按 #64279 写明 splitting ops / compile mode 与 KV 写入同图时的 allow-list 问题（注意 v0.32.0 基线自身带 #42129 allow-list）。
- [ ] ch09：同步 #64458 的 PP broadcast 宽度与 draft 通道；按 #64356 改写 batch-sharded sampling 的 owner/row（按 in-batch 位置，取代 #63554 的 `req_state_idx % TP`）。
- [ ] ch06：`AttentionBackend.supports_head_size_v`；ROCm AITER unified 可服务 V≤K；`reshape_and_cache_flash_kernel_asym`（#64064）。
- [ ] ch05：并入 offload 本批修复；DSA/DSv4.1 profile 按 #63047 计入 prefill FP32 copy（接 #62860）。
- [ ] ch02：Responses API 的 `thinking_token_budget`（#63896）。
- [ ] 模型清单（若有）：MiniMax-M3.1-Flash、Ming3-ASR；Qwen3.5 CUDA 实现路径切换。
- [ ] 更新 chapter-source-map.json 的 source_commit，跑 check_book.py 并新增阅读基线 004。
