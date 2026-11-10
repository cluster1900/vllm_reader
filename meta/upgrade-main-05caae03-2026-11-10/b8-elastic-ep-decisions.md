# 在 05caae03 基线上补写 Elastic EP 扩缩容流程与 Decisions / System One

- 结论：在本书基线 `05caae03c20f69436ff7ec23ded339ea104f1a01` 上，为第 09 章新增 §9.8.6（Elastic EP 在线扩缩 DP×EP），为第 02 章新增 Decisions / System One 短节。两处主题在 v0.33.0 已进入上游，但第一波正文只记录了 Elastic EP 的归属与校验，Decisions API 则尚未落笔。
- Reader base：`efaf5042305ee28f2bb585f2eb1ce765bafe9e0f`（B6 合入后的 `main`）。跟踪 issue：`cluster1900/bot-hub#127`。
- 上游 pin：`05caae03c20f69436ff7ec23ded339ea104f1a01`（提交日期 2026-11-10 08:54 SGT）。所有源码引用与行号均在该 commit 上人工核对。
- 章节仍为 draft，`runtime_verified=false`；全部静态源码核对，没有跑 GPU 或多机扩缩容。
- 本批只改 `book/02-vllm-architecture/README.md`、`book/09-distributed/README.md`、`meta/chapter-source-map.json` / `.md` 中的 ch02/ch09 条目，以及本说明。未改 README、`source-revisions.md` 和其他章节。
- 第 03 章 §3.4.4 现有句子「Elastic EP 的完整流程不在本章展开。」仍然正确，但没有点名 §9.8.6；本批按并行约束未改 ch03。

## 一、第 09 章 §9.8.6 Elastic EP

上游关键提交（都是 `05caae03` 的祖先，且已在 v0.33.0 中）：`dea268336`（Milestone 2，#34861）、`dd9e07a25`（#43202，external LB）。

正文覆盖：

- HTTP：`POST /scale_elastic_ep`（`api_router.py:38`，`new_data_parallel_size` 为正整数，`drain_timeout` 默认 120 秒）、`POST /is_scaling_elastic_ep`（`:84`）。
- `AsyncLLM.scale_elastic_ep`（`async_llm.py:1321`）：prepare → 按需排空 → commit。排空受 `VLLM_ELASTIC_EP_DRAIN_REQUESTS` 控制，默认关闭。
- 内部 LB（`DPLBAsyncMPClient.prepare_elastic_ep`）要求 `data_parallel_backend == "ray"`（`core_client.py:2013-2015`）。外部 LB 走 `ExternalElasticEPScaleCoordinator`（`:205`），且需 `enable_elastic_ep`、非 hybrid LB（第 09 章原有段落已述）。
- 状态机（`ElasticEPScalingState`，`:67`）：existing 扩容 `PREPARE → SYNC_KV_CACHE_MEMORY_SIZE → COMMIT_SCALE_UP → COMPLETE`；new engine `PRE_KV_INIT → PREPARE → COMPLETE`（`commit_scale_up(False)`，不等待 `commit_requested`）；scale-down remaining `PREPARE → COMMIT_SCALE_DOWN → COMPLETE`；removing `PREPARE → COMPLETE`，发 `SHUTDOWN_COMPLETE`，外部 LB 下还等 `shutdown_complete` 键（`elastic_state.py:255-278`）。
- Engine：`reinitialize_distributed`（`core.py:2870`）创建状态机；`commit_prepared_elastic_ep`（`:2924`）只把 `commit_requested` 置位。Worker 侧执行器是 `ElasticEPScalingExecutor`（`elastic_execute.py:153`）。
- 退休 rank：`set_elastic_ep_rank_retired(True)`（`async_llm.py:1363-1369`），`ScalingMiddleware`（`middleware.py:35`）对非 `/health`、`/is_scaling_elastic_ep` 的请求返回 503。
- 约束：`enable_elastic_ep` 时 `_resolve_extensible_kv_cache` 关闭可扩展 KV cache（`vllm/config/vllm.py:3204-3208`），因为扩容跳过 warmup、复用已 profile 的 KV 尺寸。可扩展 KV 本身的生命周期留给并行的第 05 章批次。
- 一张 mermaid 图（existing rank 扩容链）加读图方法；明确不把 new/existing 画成同一条状态链。没有第二张 external-LB 握手图。

§9.8 末原「留给后续批次补写」已改为指向 9.8.6。章末钉扎说明并未点名 Elastic EP，故未改。

## 二、第 02 章 Decisions / System One

上游关键提交（都是 `05caae03` 的祖先，且已在 v0.33.0 中）：`c923e7769`（#60465，`/v1/decisions` MVP）、`d33872dfe`（#59299，`/v1/systemone`）、`7e3d55956`（#62437，DiffusionGemma joint reads）。

正文覆盖：

- 注册：`vllm/entrypoints/launchers/api_server/routers.py:28-38`。
- `POST /v1/decisions`：`create_decision`（`openai/decisions/api_router.py:40`）、`OpenAIServingDecisions.create_decisions`（`serving.py:55`）；问题类型 `predicate` / `choice` / `score`（`protocol.py:59-91`）；文本 only。没有 handler 时返回 501。OpenAI 入口只在策略任务为 `generate` 时挂上。
- `POST /v1/systemone`：`ServingStructuredDecisions`（`generate/structured_decisions/serving.py:56`）。`generate` 策略用回复首 token 的标签 logprobs 再做 softmax，并要求 `--logprobs-mode raw_logprobs|processed_logprobs`（`strategies.py:1483-1486`）。`classify` / `token_classify` 专用头不在短节展开，指向 `docs/models/structured_decisions_models.md`。
- `joint=true` 只有声明 `joint_reads` 的策略（如 DiffusionGemma）走 `read_joint`，否则 400。
- 概率受 batch 同伴与 prefix cache 影响；可复现时官方文档提示 `VLLM_BATCH_INVARIANT=1`。
- 一张 mermaid 图加读图方法；softmax 有一句白话解释。

落点在 §2.3「两条路径的共享与分叉」之后、§2.4 之前，避免把 Decisions 写成与离线 / chat 并列的第三条主链。源码索引与本章小结各补了一处指向。

## 三、未写入或未改动的事项

- 未改第 03 章指针（见上文）。
- 未展开 external LB 的 `ExternalElasticEPScalePhase` 握手时序、`can_reuse_fused_moe_kernel` 的 wait/keep 细节（后者仍以 §9.8 原有段落为准）。
- 未在本节展开可扩展 KV cache 的生命周期（并行第 05 章批次）。
- 未给 `/v1/systemone` 贴长 JSON / curl 示例；未下钻 ClefOmni 等多模态 decisions 实现。
- 未改 watchdog、XPress/FLy、默认 `block_size=64`（分属其他批次）。
