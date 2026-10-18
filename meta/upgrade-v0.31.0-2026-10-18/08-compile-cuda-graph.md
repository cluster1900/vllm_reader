# 第08章：v0.31.0 迁移复核

目标源码：db9527a46873454610df6dbedf79a36d6bf1a7f6（v0.31.0）；原文行号为 original book/08-compile-cuda-graph/README.md。

升级影响判定：有实质影响。继承旧基线 mixed FULL+NEVER 报错/后续检查图修正，随后修改正文实际运行条件，不追加孤立 changelog。

| 原文位置 | 漂移/问题 | 目标源码证据 | 修正 |
|---|---|---|---|
| L55 / frontmatter | 基线需同步 | 固定目标 SHA | 正文/frontmatter 改目标 SHA、v0.31.0、2026-10-18，draft/runtime false 保留 |
| L262 | max_concurrent_batches 分支不代表 MRV1 async+PP 受支持 | vllm/config/vllm.py L592–602、L3115–3133 | 明确该组合在配置层拒绝 |
| L388 | eligible step 固定 pp_size 表述漏 XPU 例外，spec placeholders 不再固定长度 | vllm/v1/core/sched/async_scheduler.py L14–60 | 写 decode_stagger、XPU 开关例外与 per-step spec 数量 |
| L543 | AsyncOutput 不再直接保存 routed_experts | vllm/v1/worker/gpu/async_utils.py L125–205 | 说明 pending_aux_output 排队/完成的生命周期 |
| L1305、L1313–1323 | 有 max_query_len 约束的图须拒绝未知 max_query；短 prefill 不能误匹配 varlen decode graph | vllm/v1/worker/gpu/cudagraph_utils.py L103–129；gpu/model_runner.py L1711–1718 | 补条件与图中拒绝分支 |
| L1327–1341 | DBO 微批次 graph 候选现有真实生成/捕获逻辑，不能只列描述符字段 | vllm/v1/worker/gpu/cudagraph_utils.py L222–246、L391–488 | 说明 FULL-only twin、uniform/threshold 限制、capture stream 与失败清理 |
| L1382–1384、L1421–1437 | 缺少 compiled/breakable provider 时不再正常直到 capture 才报错 | vllm/v1/worker/gpu/model_runner.py L701–718；vllm/config/compilation.py L80–85、L1464–1477；tests/test_config.py L908 | 重写正常解析 fallback，保留 capture 防御性报错；图补 without_piecewise 分支 |

## 保留内容与地图

三种优化轴、配置五模式/运行三模式、compile cache、MRV1 keys、静态 buffer、piecewise 先于 full capture、graph-memory 预算主干仍成立。现有 source-map 路径与核心类保持有效；可在 compilation.py 锚点增补 CUDAGraphMode.without_piecewise，在 config tests 锚点增补 test_resolve_cudagraph_mode_uses_loaded_piecewise_provider。

## 验证边界

2026-10-18 静态源码复核；未安装/运行锁定上游的 GPU runtime，未执行模型加载、CUDA capture/replay、NCCL、D2H trace 或上游 GPU correctness tests。章节保持 draft / runtime_verified: false。

在对应工作副本运行 `python3 -m unittest tests.test_ch07_execution_pipeline tests.test_ch08_async_compile_graph -v`：33 tests passed。两个对应 CPU examples 均退出 0。这些测试只验证随书教学契约，不覆盖本轮新增的上游配置分支。
