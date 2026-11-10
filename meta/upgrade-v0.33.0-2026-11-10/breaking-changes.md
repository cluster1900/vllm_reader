# v0.32.0 → v0.33.0：破坏性变更与弃用清单

- 日期：2026-11-10；静态源码核对，不含运行时验证。
- 新基线：v0.33.0 = `4ba787ec9959e566d3b587a796dff07673324da5`；旧基线：v0.32.0 = `5df63a7203cdb9328d824a27b682e77c7b9928be`。
- 来源：v0.33.0 release notes 的 “Breaking Changes & Deprecations” 一节（下称“release notes”），加上本批次（ch01/02/06/10）核对源码时发现、release notes 没有列出的行为变化。
- 祖先关系证明：v0.32.0 是 release 分支上的提交，不是 v0.33.0 的祖先（两者的 merge-base 为 `fe341a53`），所以不能用 `5df63a72..4ba787ec` 的提交链判断某个改动是否进入 v0.33.0。本清单对**每一条**引用的提交单独执行
  `git merge-base --is-ancestor <commit> 4ba787ec`，下表“祖先”列为 `IN` 表示命令返回 0。
- 范围：只列 v0.33.0 祖先链里的改动。不在祖先链里的 main 变化（例如 [#61258](https://github.com/vllm-project/vllm/pull/61258)、#63633、#64064、#64097、#64177、#63896）仍留在各章的“main 上的变化（未发版）”注记中，等待下一波次。
- 本清单只覆盖本批次章节（01、02、06、10）与它们依赖的行为；其他章节的条目由对应批次的 PR 处理，下表“本书章节”列写明归属。

## A. release notes 列出的破坏性变更与弃用

| # | 变更 | PR / 提交 | 祖先 | 本书章节与处理 |
|---|---|---|---|---|
| A1 | 发布 wheel 与镜像只剩 CUDA 13.0，CUDA 12.9 产物不再发布 | [#62575](https://github.com/vllm-project/vllm/pull/62575) / `083d61622` | IN | 本书没有引用；ch10 §10.10.1 “固定环境”清单本来就要求记录 CUDA 版本，无需改文。 |
| A2 | 移除 Medusa 与 MLP speculator | [#61701](https://github.com/vllm-project/vllm/pull/61701) / `8530aed2b` | IN | 全书没有出现这两个名字，无需改文。 |
| A3 | 移除旧 Ray compiled-graph executor 与 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND`；`RayExecutorV2` 是唯一 Ray executor | [#61712](https://github.com/vllm-project/vllm/pull/61712) / `e57bb4ace`、[#62228](https://github.com/vllm-project/vllm/pull/62228) / `52e88e25e` | IN | ch07（执行器选择）与 ch08、ch09 的 Ray 论述涉及，由对应批次处理；ch01/02/06/10 不涉及。 |
| A4 | 弃用 `--mamba-cache-mode`（已无效果） | [#60043](https://github.com/vllm-project/vllm/pull/60043) / `2cc2e68d1` | IN | 本批章节没有引用。 |
| A5 | 弃用 `--ir-op-priority`（计划 v0.34.0 移除） | [#63371](https://github.com/vllm-project/vllm/pull/63371) / `a50259bb3` | IN | 本批章节没有引用。 |
| A6 | 弃用 LM Format Enforcer structured-output backend（v0.34.0 起需显式开启，v0.35.0 移除） | [#63423](https://github.com/vllm-project/vllm/pull/63423) / `4e5916571` | IN | 本批章节没有引用。 |
| A7 | **生成暂停期间到达的新请求返回 HTTP 503**（`abort`、`wait` 模式；`keep` 模式仍排队） | [#51488](https://github.com/vllm-project/vllm/pull/51488) / `d5b36699b` | IN | **ch02 §2.14 “暂停期间到达的新请求会被拒绝”** 与 §2.11 新增分支，含时序图；调用链 `EngineCoreProc.pause_generation`（`vllm/v1/engine/core.py:2225`）→ `_reject_add_while_paused`（`:1796`）→ `OutputProcessor.process_outputs` error 分支（`vllm/v1/engine/output_processor.py:764`）→ `RetryableRequestError`（`vllm/exceptions.py:139`，`status_code=503`）。 |
| A8 | `kv_transfer_params` / `ec_transfer_params` 移到 render 输出的顶层 | [#61241](https://github.com/vllm-project/vllm/pull/61241) / `c6b772cf6` | IN | ch02 没有对应论述，无需改文。 |
| A9 | 给不支持 LoRA 的模型发 LoRA 请求，在校验阶段失败 | [#61884](https://github.com/vllm-project/vllm/pull/61884) / `37891a98e` | IN | 本批章节没有引用。 |
| A10 | RL 权重传输 API 改用 `ParamMeta`、`iter_full()`、`iter_local()` | [#63256](https://github.com/vllm-project/vllm/pull/63256) / `50a184f43` | IN | 本书没有引用。 |
| A11 | beam search 分数按生成长度归一化 | [#62928](https://github.com/vllm-project/vllm/pull/62928) / `7d5335936` | IN | ch02 只在范围声明里提到 beam search，没有分数公式，无需改文。 |
| A12 | Rust 前端：不支持的 `vllm serve` 参数与请求字段改为失败；请求自带 chat template 需要 `--trust-request-chat-template`；空 `tools` 数组返回 400 | [#62431](https://github.com/vllm-project/vllm/pull/62431) / `552868555`、[#62434](https://github.com/vllm-project/vllm/pull/62434) / `255e3a310`、[#62673](https://github.com/vllm-project/vllm/pull/62673) / `29a26e5b0`、[#63060](https://github.com/vllm-project/vllm/pull/63060) / `654bd123b` | IN | 本书只描述 Python 前端，无需改文。 |
| A13 | NIXL connector 需要 NIXL 1.5 或更新版本 | [#61032](https://github.com/vllm-project/vllm/pull/61032) / `eaf25f675` | IN | ch05、ch07 的 KV connector 论述涉及，由对应批次处理。 |
| A14 | 移除环境变量 `VLLM_ROCM_AITER_MLA_DCP_VERIFY`、`VLLM_ROCM_USE_AITER_FP4_ASM_GEMM`、`VLLM_ROCM_AITER_MEGA_MOE_FUSE_SHARED_EXPERTS`；`VLLM_ROCM_MONO_DECODE` 由 `--monokernel` 取代 | [#61628](https://github.com/vllm-project/vllm/pull/61628) / `39504080e`、[#61278](https://github.com/vllm-project/vllm/pull/61278) / `2067297b2`、[#61345](https://github.com/vllm-project/vllm/pull/61345) / `ffa18e55a`、[#61082](https://github.com/vllm-project/vllm/pull/61082) / `cc2a6eea1` | IN | 全书没有引用。 |
| A15 | 新默认值：FA4 sparse MLA decode 在 Blackwell 上对每 rank ≤16 个 query head 启用 | [#55866](https://github.com/vllm-project/vllm/pull/55866) / `31e90ac37` | IN | **ch06 §6.15.4**：CUDA 候选表里 major 10、sparse MLA、BF16 KV、head ≤16 时 `FLASH_ATTN_MLA_SPARSE_FA4` 排在 FlashInfer/FlashMLA 之前。 |
| A16 | 新默认值：EAGLE/MTP 的 lookahead block hash 默认开启 | [#61589](https://github.com/vllm-project/vllm/pull/61589) / `2fb320d83` | IN | ch04、ch05 的 lookahead 论述涉及，由对应批次处理；ch01/02/06/10 没有引用。 |
| A17 | 其余“New defaults”（GLM-5.3-Flash FP8 KV、Cohere2MoE dual stream、ROCm AITER/FlyDSL 项） | [#61228](https://github.com/vllm-project/vllm/pull/61228) / `2f18d13cf`、[#62101](https://github.com/vllm-project/vllm/pull/62101) / `0d34366b1`、[#53123](https://github.com/vllm-project/vllm/pull/53123) / `81503550f`、[#60645](https://github.com/vllm-project/vllm/pull/60645) / `f9b65c30c`、[#56226](https://github.com/vllm-project/vllm/pull/56226) / `47fd76e4c`、[#60250](https://github.com/vllm-project/vllm/pull/60250) / `9da5f2d31`、[#63669](https://github.com/vllm-project/vllm/pull/63669) / `d0d681e40` | IN | 与本批章节论述无关，无需改文。 |

## B. release notes 没有列出、但会让旧正文失效的行为变化（本批章节）

这些条目都已用源码核对，并在对应章节改了正文。

| # | 变化 | PR / 提交 | 祖先 | 本书章节与处理 |
|---|---|---|---|---|
| B1 | **chat 基准的 TTFT/ITL 判定收紧**：只有 `delta` 带非空 `content`/`reasoning`/`tool_calls`/`function_call`，或 `content==""` 且无 `role` 的消息才算已流出输出；只有 `role` 的开头消息不再触发 TTFT。跨版本比较 chat TTFT 时口径不同。 | [#62341](https://github.com/vllm-project/vllm/pull/62341) / `dcf8158cd` | IN | **ch10 §10.2.3**（含判定流程图）。 |
| B2 | `bench serve` 的 probe 请求改用独立单连接 session；`benchmark_duration` 先于停止 probe 计算（此前最后一个 probe 间隔会计入主 workload 吞吐分母） | [#60511](https://github.com/vllm-project/vllm/pull/60511) / `c54dca96b` | IN | **ch10 §10.7.1、§10.13.4**（含时序图）。 |
| B3 | `bench serve` 不再对每个网络字节块 `strip()`，避免空白在传输分块边界被吞掉 | [#51172](https://github.com/vllm-project/vllm/pull/51172) / `02d69c663` | IN | **ch10 §10.2.3**。 |
| B4 | `bench serve` 结果新增逐请求 `server_queue_times`、`server_ttfts`（需要 `--enable-per-request-metrics` 与流式 usage） | [#55809](https://github.com/vllm-project/vllm/pull/55809) / `ae43d045c` | IN | **ch10 新增 §10.11.5**。 |
| B5 | `/tokenize` 对齐探测失败时的警告区分连接失败 / 404 / 其他状态码 / 其他异常 | [#59888](https://github.com/vllm-project/vllm/pull/59888) / `bafb26366` | IN | **ch10 §10.8.4**。 |
| B6 | `/start_profile` 请求体新增 `profiler_kwargs`；`Worker.profile`、`TorchProfilerWrapper.set_profiler_kwargs`、`LLM.start_profile` 同步新增 | [#51974](https://github.com/vllm-project/vllm/pull/51974) / `dd5e75449` | IN | **ch10 §10.15.2**（含数据流图）。 |
| B7 | Worker 在内存快照前创建 cuBLAS/cuBLASLt handle，使其计入显存基线 | [#61044](https://github.com/vllm-project/vllm/pull/61044) / `9cede34a6` | IN | **ch10 §10.9.1**。 |
| B8 | 滑窗 group 的 block table 在 gather 时把窗口之外的整块表项换成 null block（`num_computed_tokens` 新参数） | [#60767](https://github.com/vllm-project/vllm/pull/60767) / `e1f314164` | IN | **ch06 新增 §6.6.4**（含公式、手算例子与流程图）。 |
| B9 | `BlockTables` 记录 `_num_dirty_rows`，共享缓冲里上一批更大 batch 留下的旧行也被清零 | [#61174](https://github.com/vllm-project/vllm/pull/61174) / `61bb59087` | IN | **ch06 §6.6.3**。 |
| B10 | staged write kernel 的 grid 变成 `(写入条数, ceil(最长/1024))`，每个 program 复制 1024 个元素 | [#62698](https://github.com/vllm-project/vllm/pull/62698) / `fc2b6ab45` | IN | **ch06 新增 §6.6.5**（含公式与流程图）。 |
| B11 | `AttentionBackend.supports_kv_cache_update_bypass` 与 `Attention.kv_cache_update_bypass` 新增；FlashAttention 后端声明 `True` | [#62622](https://github.com/vllm-project/vllm/pull/62622) / `6ab6bff1b`（框架）、[#61879](https://github.com/vllm-project/vllm/pull/61879) / `0f6b26191`（FA 取值） | IN | **ch06 §6.13、§6.16.1**。 |
| B12 | `FlashAttentionMetadataBuilder._cudagraph_support`：FA4 与 XPU 也是 `ALWAYS`（FA3 在 v0.32.0 已是 `ALWAYS`）；XPU 平台 head size 上限 512 | [#60908](https://github.com/vllm-project/vllm/pull/60908) / `791f92135`（FA4）、[#62244](https://github.com/vllm-project/vllm/pull/62244) / `35d01660f`（XPU）、[#62014](https://github.com/vllm-project/vllm/pull/62014) / `d6d5412ca`（XPU head size） | IN | **ch06 §6.16.1、§6.16.2**。 |
| B13 | multimodal prefix range 计算移到 `vllm/v1/attention/backends/utils.py::compute_mm_prefix_ranges`，MRV2 专用副本删除 | [#56937](https://github.com/vllm-project/vllm/pull/56937) / `b3b0bc305` | IN | **ch06 §6.10**。 |
| B14 | 可插拔 KV cache dtype：`register_kv_cache_dtype`、`KVCacheDTypeHandler`、`is_known_kv_cache_dtype`；`is_quantized_kv_cache` 从 `vllm/utils/torch_utils.py` 移到 `vllm/v1/kv_cache_interface.py` | [#52670](https://github.com/vllm-project/vllm/pull/52670) / `9499c949b` | IN | **ch06 §6.15.5**、**ch01 KV 容量公式后的说明**。 |
| B15 | MRV2 小 batch split top-p：新增 `has_topp_only_rows` 判定，混合 batch 可跳过补跑；split top-p 的并列配额跨 block 累计，logits 极大时不再屏蔽整行 | [#61996](https://github.com/vllm-project/vllm/pull/61996) / `7b6310c10`、[#62046](https://github.com/vllm-project/vllm/pull/62046) / `ce49c47cb`、[#58231](https://github.com/vllm-project/vllm/pull/58231) / `852989f58` | IN | **ch01 “Top-k 与 Top-p”**（含公式、手算例子与决策图）。 |
| B16 | chat 流式输出拆成 `chat_completion_stream_generator_raw`（产出对象）与外层 SSE 序列化；非流式消息字段由 `assemble_chat_message_fields` 组装；Cohere/Anthropic 复用 raw 生成器 | [#62706](https://github.com/vllm-project/vllm/pull/62706) / `5ea2c7ff1`、[#63148](https://github.com/vllm-project/vllm/pull/63148) / `e879223b5`、[#61947](https://github.com/vllm-project/vllm/pull/61947) / `5be4266f5` | IN | **ch02 §2.2 在线入口**（含流程图）。 |
| B17 | 渲染阶段客户端错误从 `ValueError` 改为带 `parameter` 的 `VLLMValidationError`；`skip_tokenizer_init` 时仍按模型词表检查 token ID | [#62738](https://github.com/vllm-project/vllm/pull/62738) / `1cc332ee8`、[#62911](https://github.com/vllm-project/vllm/pull/62911) / `47e2b610b` | IN | **ch02 §2.4**。 |
| B18 | `--reasoning-effort-map` 改写请求 `reasoning_effort`；`tool_strict_level` 缺省改为回落到 tool parser 默认；回显 prompt 在截断之后再反解 | [#61094](https://github.com/vllm-project/vllm/pull/61094) / `41d9d1e14`、[#61328](https://github.com/vllm-project/vllm/pull/61328) / `8fa5ae844`、[#61876](https://github.com/vllm-project/vllm/pull/61876) / `d74f4180c` | IN | **ch02 §2.4**。 |
| B19 | 流式输入会话：collector 不跨 chunk 边界合并；ABORT/ERROR 终止会话；前端终结时通知 EngineCore | [#58941](https://github.com/vllm-project/vllm/pull/58941) / `957d512e9`、[#58942](https://github.com/vllm-project/vllm/pull/58942) / `146350155` | IN | **ch02 §2.11**（含 `process_outputs` 分支图）。 |
| B20 | input thread 把 structured-output backend 不一致等 `VLLMValidationError` 作为带错误的输出回送；多模态 cache 失配的失效标记先于 `process_outputs` 入队 | [#49625](https://github.com/vllm-project/vllm/pull/49625) / `88077b746`、[#62164](https://github.com/vllm-project/vllm/pull/62164) / `70806b631` | IN | **ch02 §2.8**。 |
| B21 | DP wave 的唤醒权交给引擎，前端不再发送 `FIRST_REQ` | [#59559](https://github.com/vllm-project/vllm/pull/59559) / `a94c31571` | IN | **ch02 §2.6 “DP 变体”**。 |
| B22 | busy loop 看门狗：每步 `feed()`，阻塞读输入时 `disable()` | [#55700](https://github.com/vllm-project/vllm/pull/55700) / `1c8799197` | IN | **ch02 §2.13**。 |
| B23 | `vllm/entrypoints/serve/utils/api_utils.py` 里的启动辅助函数（`get_non_default_args`、`log_non_default_args`、`jsonify_non_default_args`、`redact_sensitive_args`、`cli_env_setup`、`log_version_and_model`）移到 `vllm/entrypoints/common/startup.py`；该文件里的 `get_max_tokens` 保持不变 | [#62390](https://github.com/vllm-project/vllm/pull/62390) / `5caabdbb6` | IN | ch02 只引用 `get_max_tokens`，无需改文；外部代码若从旧路径导入这些函数需要改导入路径。 |
| B24 | `IterationStats.__init__` 一次性读取 `VLLM_COMPUTE_NANS_IN_LOGITS`（纯性能优化，语义不变） | [#61710](https://github.com/vllm-project/vllm/pull/61710) / `6fe7401fa` | IN | **ch10 §10.11.1**。 |

## C. 复核方法

```bash
# 对每个提交：返回码 0 即在 v0.33.0 祖先链中
git -C ../vllm merge-base --is-ancestor <commit> 4ba787ec9959e566d3b587a796dff07673324da5; echo $?

# v0.32.0 不是 v0.33.0 的祖先；merge-base 为 fe341a53
git -C ../vllm merge-base 5df63a7203cdb9328d824a27b682e77c7b9928be 4ba787ec9959e566d3b587a796dff07673324da5
```

提交与 PR 的对应取自上游 `git log --grep="(#N)"`；表中所有提交的 `--is-ancestor` 结果均为 IN。
