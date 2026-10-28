# vLLM main 跟踪记录（2026-10-27～2026-10-28，未发版）

- 当前阅读基线：`v0.32.0` / `5df63a7203cdb9328d824a27b682e77c7b9928be`（基线 003），正文不变。上游仍无新 release（最新 v0.32.0，2026-10-18）。
- 跟踪区间：`vllm-project/vllm` `main` `ed1a7307950ab5e6593d25123579ed7ca998b350`..`0106c8abce0f8fec21d2871e5b7404acd9c75065`（[#61831](https://github.com/vllm-project/vllm/pull/61831)，2026-10-28 08:54 UTC+8），共 103 个提交。接续 [2026-10-27 记录](main-tracking-2026-10-27.md)。
- 方法：在只读 partial clone 中逐条 `git show` 阅读下表提交的非测试源码 diff；只记录读过 diff 的条目。没有运行上游代码、GPU 或模型。影响表共 33 行。其余 67 个提交已按“是否修改书中 `[源码]` 锚点文件”逐条复扫：命中但不改书中描述符号的（如 `vllm/config/vllm.py` 配置边角、`vllm/v1/worker/utils.py` 的 KV 视图拷贝、`topk_topp_sampler` 的 nvcc 回退、`parallel_state`/cudagraph stream、NIXL/Mooncake connector 内部）列入“仅清单”或本节“不受影响/边角”；其余为 CI、文档、ROCm/XPU 专用、具体模型 recipe 与 AMD 镜像。
- 结论：v0.32.0 正文对 v0.32.0 本身仍成立；共加/追加“main 上的变化（未发版）”注记：ch05（prefix cache API #60960；KVCR #60469/#63889；HiSparse backend #62592）、ch07（NUMA 默认 #61258；MRV2 CUDA graph profile 门控 #63778）、ch08（同 #63778）、ch09（PCP 关闭 draft logits 分片 #64151；NUMA #61258）、ch10（实验合同中的 NUMA/`numa_bind` 默认 #61258）。其中与 v0.32.0 正文描述冲突、升级时须改写正文的是 #60960、#64151、#63778、#61258。

## 影响表

| SHA / PR | 领域 | main 上的变化 | 本书位置 | 处理 |
|---|---|---|---|---|
| [c4f2bb4abb](https://github.com/vllm-project/vllm/commit/c4f2bb4abbdbad9ed5b0a71acd0bd585056eeab8) / [#61557](https://github.com/vllm-project/vllm/pull/61557) | spec decode / structured | `StructuredOutputManager.validate_tokens` 改名为 `trim_spec_tokens`：保留 grammar-invalid 的 draft 槽位给 rejection sampling，不再在 scheduler 侧滤掉；`grammar_bitmask` 对 padding/非法 draft 继续填 bitmask 而不提前 unconstrained | 书中未展开 structured+spec 滤 token | 仅清单 |
| [820be160ce](https://github.com/vllm-project/vllm/commit/820be160ce0dc9098deaa2a4c6b5fc534c43881d) / [#64151](https://github.com/vllm-project/vllm/pull/64151) | spec decode / PCP | `Speculator` 开启 batch-sharded draft logits 时新增条件 `prefill_context_parallel_size == 1`：PCP 开启时不分片，因 `PCPManager.sync_drafts` 按 request state 整行索引 draft logits（`speculator.py` 约 L282–L287 @ `0106c8abc`） | ch09 batch-shard / §9.9.3 | 注记（与 #63554 注记叠加） |
| [43111fd1f0](https://github.com/vllm-project/vllm/commit/43111fd1f0c3135ff671e25e2e85238aa05743d0) / [#53104](https://github.com/vllm-project/vllm/pull/53104) | spec decode（EAGLE） | `_should_share`：若 draft/target 的 `weight` 不是 Tensor（量化层用 `weight_packed`/`qweight`），不再 `torch.equal`，返回 False 保留 draft 自己的 lm_head，避免误共享量化头 | 书中未展开 EAGLE 共享 lm_head | 仅清单 |
| [e7fa26d5c8](https://github.com/vllm-project/vllm/commit/e7fa26d5c8d94636a4d0525e1a2462cb661ea331) / [#63996](https://github.com/vllm-project/vllm/pull/63996) | spec decode（DeepSeek） | 抽出 `needs_mtp_hidden_buffer`（DSpark 跳过 MTP hidden buffer）；`CommonAttentionMetadata.invalidate_cached_layout` 在 in-place 改 `seq_lens` 后丢弃缓存布局；tilelang MHC kernel 改进 | 书中未展开 #41834 / DSpark buffer | 仅清单 |
| [c6f6e6a983](https://github.com/vllm-project/vllm/commit/c6f6e6a9835c3e4df9c06bb6c564c7d866be9f22) / [#63841](https://github.com/vllm-project/vllm/pull/63841) | spec decode（DSpark） | `GlmMoeDsaForCausalLM`：draft 方法为 `dspark`（与 `mtp` 一样）时不再把未设置的 `spec_config.kv_cache_dtype` 强制为 `auto`，从而可沿用目标的 fp8 KV cache（声称每 rank 约省 5.38 GB） | ch09 DSpark 未写 dtype | 仅清单 |
| [e1e77a279d](https://github.com/vllm-project/vllm/commit/e1e77a279d68a9601be062cb267a5f9911c422c7) / [#60960](https://github.com/vllm-project/vllm/pull/60960) | prefix cache | `Request.get_skip_reading_prefix_cache` 改为 `get_prefix_cache_hit_limit` / `get_max_cache_hit_length`：`prefix_cache_lookup_enabled` 看 `prefix_cache_hit_limit != 0`；有 `prompt_logprobs`/`prompt_logprob_token_ids` 时上限为 `prompt_logprob_start or 0`（显式 `skip_reading_prefix_cache=False` 时可复用评分起点之前的本地前缀）；一旦 `prefix_cache_hit_limit is not None` 则 `skip_reading_external_kv` 为 True，调度器跳过外部 KV connector 命中（`scheduler.py` L1132–L1133），仅允许有界的本地前缀复用 | ch05 §5.12.1–5.12.2 | 注记（与正文冲突） |
| [e3883b7a7e](https://github.com/vllm-project/vllm/commit/e3883b7a7ec41f3117dfb7ca764746ec80a24d87) / [#60469](https://github.com/vllm-project/vllm/pull/60469) | KV offload（KVCR） | KVCR manager 在 store 完成时按请求聚合 `completed_prefixes`，跨共享前缀 key 保序，由 scheduler finalize 请求 recency，不再对已完成前缀逐块 touch | ch05 §5.15.4 | 注记（汇总） |
| [36f5ff3172](https://github.com/vllm-project/vllm/commit/36f5ff3172eb75433e54e496d64c473e951f8ecc) / [#63889](https://github.com/vllm-project/vllm/pull/63889) | KV offload（KVCR） | 降低 KVCR adapter 开销（framework pin / key adapter 路径整理，少一次多余遍历） | ch05 §5.15.4 | 注记（汇总） |
| [4d95f222e6](https://github.com/vllm-project/vllm/commit/4d95f222e68bd43acbdfdffad6d46eea70146dcc) / [#61258](https://github.com/vllm-project/vllm/pull/61258) | Core / NUMA | `ParallelConfig.numa_bind` 默认由 `False` 改为 `None`：环境允许时 best-effort 自动绑定（软 mem preference、`_numa_bind_auto`）；`True` 仍为严格失败即退出，`False` 关闭。`numa_utils.configure_subprocess` 在 `None` 时探测 | ch07 Worker 启动；ch09 ParallelConfig；ch10 NUMA 检查项 | 注记 |
| [dc4d977d97](https://github.com/vllm-project/vllm/commit/dc4d977d97f9cfaa7c6dc2de7fd94a77204a6914) / [#63778](https://github.com/vllm-project/vllm/pull/63778) | MRV2 / CUDA Graph | `Worker.determine_available_memory`：V2 runner 时只要平台 CUDA-alike/XPU 就调用 `model_runner.profile_cudagraph_memory()`，不再在 Worker 侧要求 `cudagraph_mode!=NONE` 或展开 `cudagraph_mm_encoder` 条件（由 runner 自决，因 encoder graph 不依赖 cudagraph_mode） | ch07 §7.9–7.10；ch08 §8.17 | 注记（与正文冲突） |
| [939ff73503](https://github.com/vllm-project/vllm/commit/939ff73503a47c7acae280f095b7f16fb7fcf5fa) / [#63785](https://github.com/vllm-project/vllm/pull/63785) | model_executor | 删除 `BasevLLMParameter.__torch_function__` | 书中未展开 Parameter 包装 | 仅清单 |
| [70cb50dab2](https://github.com/vllm-project/vllm/commit/70cb50dab240a03f7db9ab87a613b5b52f1339fe) / [#62592](https://github.com/vllm-project/vllm/pull/62592) | HiSparse | 新增 `vllm/v1/hisparse/backend.py` 的 `HiSparseBackend`（pin/unpin、cache ops、IPC event、`prepare_kv_cache_views`），CUDA 平台注册；runtime/connector 改走 backend | ch05 HiSparse 段（未写 backend 抽象） | 注记 |
| [2b08392a5f](https://github.com/vllm-project/vllm/commit/2b08392a5f) / [#63853](https://github.com/vllm-project/vllm/pull/63853) | spec decode | draft `max_model_len` 限制移到 model config hooks 之前 | 无章节展开 | 仅清单 |
| [8701c9b751](https://github.com/vllm-project/vllm/commit/8701c9b751) / [#55112](https://github.com/vllm-project/vllm/pull/55112) | KV copy | `copy_kv_cache_blocks_inplace` 拷贝每一个 overlay 的 KV view | 书中未写该 helper | 仅清单 |
| [cebcc5d8c7](https://github.com/vllm-project/vllm/commit/cebcc5d8c7) / [#63548](https://github.com/vllm-project/vllm/pull/63548) | MRV2 sleep | sleep mode 下 MRV2 model state 留在 runtime pool | ch08 sleep/cuMem 未细化 | 仅清单 |
| [05a303fdc8](https://github.com/vllm-project/vllm/commit/05a303fdc8) / [#62570](https://github.com/vllm-project/vllm/pull/62570) | Frontend / Engine | checkpoint lifecycle 的 HTTP prepare/status/resume | 书中未讲 checkpoint API | 仅清单 |
| [274a5e19a1](https://github.com/vllm-project/vllm/commit/274a5e19a1) / [#63155](https://github.com/vllm-project/vllm/pull/63155) | DSv4.1 / graph | decoder replay graphs 捕获到 model graphs 的 stream | ch08 capture 未写 DSv4.1 replay | 仅清单 |
| [4ddc7aa40f](https://github.com/vllm-project/vllm/commit/4ddc7aa40f) / [#62317](https://github.com/vllm-project/vllm/pull/62317) | NIXL | hybrid SSM transfer 只读本组 region | ch09 connector 未展开 | 仅清单 |
| [2897401370](https://github.com/vllm-project/vllm/commit/2897401370) / [#62478](https://github.com/vllm-project/vllm/pull/62478) | Mooncake | 向量化 transfer descriptor 构建 | 同上 | 仅清单 |
| [3bc521d9e6](https://github.com/vllm-project/vllm/commit/3bc521d9e6) / [#61543](https://github.com/vllm-project/vllm/pull/61543) | HiSparse / connector | DecodeBenchConnector 在 bounded KV group 上的崩溃修复 | 无章节提及 | 仅清单 |
| [803bc13cf9](https://github.com/vllm-project/vllm/commit/803bc13cf9) / [#62312](https://github.com/vllm-project/vllm/pull/62312) | Rust Frontend | `--reasoning-effort-map` | 书中未讲 Rust 前端 | 仅清单 |
| [4216a234cf](https://github.com/vllm-project/vllm/commit/4216a234cf) / [#61877](https://github.com/vllm-project/vllm/pull/61877) | Rust Frontend | per-request speculative decoding metrics | 同上 | 仅清单 |
| [447d28eef2](https://github.com/vllm-project/vllm/commit/447d28eef2) / [#63292](https://github.com/vllm-project/vllm/pull/63292) | Rust Frontend | prompt-cache usage / `created_cache_tokens` | 同上 | 仅清单 |
| [7fe793a5d5](https://github.com/vllm-project/vllm/commit/7fe793a5d5) / [#63769](https://github.com/vllm-project/vllm/pull/63769) | Rust Frontend | `--enable-log-outputs` / `--enable-log-deltas` | 同上 | 仅清单 |
| [ace90aeb24](https://github.com/vllm-project/vllm/commit/ace90aeb24) 等 Rust Frontend 一批（#63762–#63772、#63789、#63707、#63709、#60856） | Rust Frontend | DeepSeek V4 content blocks、completions null/text_offset、metrics、Gemma 4 tools、`/server_info`、kv_transfer_params、profiler_kwargs、logprobs 顺序、gRPC 跳过 detok、fastokens、Lark、NIXL stats 解码 | 书中未讲 | 仅清单（合并） |
| [c325754771](https://github.com/vllm-project/vllm/commit/c325754771) / [#60514](https://github.com/vllm-project/vllm/pull/60514) | sampling | 缺 nvcc 时 DeepGEMM/FlashInfer sampling 回退 | ch01/采样未依赖该路径 | 仅清单 |
| [741584f8ca](https://github.com/vllm-project/vllm/commit/741584f8ca) / [#58744](https://github.com/vllm-project/vllm/pull/58744) | MLA | NVFP4 chunked-context gather | 无章节展开 | 仅清单 |
| [0855defde8](https://github.com/vllm-project/vllm/commit/0855defde8) / [#64085](https://github.com/vllm-project/vllm/pull/64085) | NIXL | 回退 NIXL 1.5.1 升级 | 依赖版本边角 | 仅清单 |
| [f81f0f2509](https://github.com/vllm-project/vllm/commit/f81f0f2509) / [#63427](https://github.com/vllm-project/vllm/pull/63427) | CUDA Graph | 对 captured graph 的 nested outputs 使用 weak-ref | ch08 未写 nested output 持有 | 仅清单 |
| [8977b13a1c](https://github.com/vllm-project/vllm/commit/8977b13a1c) / [#59986](https://github.com/vllm-project/vllm/pull/59986) | Attention | DSV4.1 sparse prefill 不进 FULL cudagraph | ch08/ch09 未写该禁令细节 | 仅清单 |
| [8697bbae4c](https://github.com/vllm-project/vllm/commit/8697bbae4c) / [#59392](https://github.com/vllm-project/vllm/pull/59392) | MLA / DCP | FlashInfer MLA DCP decode 在 ragged spec-decode batch 上的修复 | 书中未展开 | 仅清单 |
| [b4a0830f66](https://github.com/vllm-project/vllm/commit/b4a0830f66) / [#62464](https://github.com/vllm-project/vllm/pull/62464) | serialization | 序列化路径去掉多余 tensor 操作 | 无章节提及 | 仅清单 |
| [2fb083…略] 其余模型/ROCm/CI/recipe | 多类 | MiniMax/Kimi/Qwen/ROCm aiter、CI 自动分片、文档 recipe 等 | 多数章节不受影响 | 一句带过 |

## 不受影响（本区间相对书中锚点）

- ch01–ch04、ch06：本区间无改写其 v0.32.0 正文锚点符号的提交；ch06 staged-write 正文仍以既有 #63633 注记为准，本区间无新冲突。ch10 仅在实验合同的 NUMA 检查项旁追加 #61258 注记（不改 v0.32.0 正文锚点）。
- ch02 生命周期总览、ch04 调度主路径：无直接符号变更。
- Rust Frontend 全批次：书中不覆盖，升级清单不要求改章节正文。

## 升级清单（下一个 tag 发版后）

- [ ] 确认新 tag 是否包含上表各提交。
- [ ] ch05 §5.12：按 #60960 改写 prefix cache 读取控制（`get_prefix_cache_hit_limit` / `get_max_cache_hit_length`；`prompt_logprobs` 默认仍可跳过，但可用 `skip_reading_prefix_cache=False` + `prompt_logprob_start` 限定本地复用长度）；更新源码锚点。
- [ ] ch05 §5.15.4：并入 KVCR #60469/#63889；HiSparse #62592 backend 抽象。
- [ ] ch07 / ch09 / ch10：按 #61258 写明 `numa_bind` 默认 `None` 的 best-effort 行为与严格 `True`/`False` 差异。
- [ ] ch07 / ch08 §8.17：按 #63778 改写“非 NONE 才 profile”的 Worker 侧条件（V2 由 runner 决定）。
- [ ] ch09：按 #64151 在 batch-sharded draft logits 条件中写明 PCP 关闭分片。
- [ ] 更新 chapter-source-map.json 的 source_commit，跑 check_book.py 并新增阅读基线 004。
