# 第07章：v0.31.0 迁移复核

目标源码：db9527a46873454610df6dbedf79a36d6bf1a7f6（v0.31.0）。原文行号对应 original book/07-model-execution/README.md。

升级影响判定：有实质影响，不能只改版本头。已继承第07章旧基线三项事实/图示修正，并在原节内更新下列行为。

| 原文位置 | 漂移/问题 | 目标源码证据 | 修正 |
|---|---|---|---|
| L48 / frontmatter | 全文仍绑定旧基线 | 已固定目标 SHA | frontmatter 与正文同步 SHA、branch=v0.31.0、verified_at=2026-10-18；保留 draft/runtime false |
| L638–645 / §7.12.1 | 原默认 GPU base 搬运仍成立，但 UVA 变体增加平台 fallback 和内容 UVA | vllm/v1/worker/gpu/buffer_utils.py L151–175、L196–219 | 保留默认路径图；明确 UVA 条件和变体写入内容不必 H2D，限定图适用范围 |
| L783–786、L908–920、L937 | ExecuteModelState / output 中 routed_experts 字段已移除 | vllm/v1/worker/gpu/model_runner.py L1695–1699、L2138–2151、L2367–2376；vllm/v1/outputs.py L260–306；vllm/v1/worker/gpu/async_utils.py L202–205 | 修正临时状态与输出表；加入 begin_step → prepare_output → async copy → process_output 辅助结果链 |
| L846–866 | 采样链新增 custom processors，fused 路径资格与 XPU 路径未写入 | vllm/v1/worker/gpu/sample/sampler.py L65–100、L223–360 | 重写采样顺序及图：native/custom chain → thinking budget → temperature/min-p → fused 或显式过滤 Gumbel |
| L979–1004 / §7.22 | env 并非最高优先级；ROCm 默认受 MRV1 能力约束；custom processor、KV fast prefill 已不在 V2 fallback 清单 | vllm/config/vllm.py L701–763、L3051–3163；tests/test_config.py L830、L3714 | 重写选择图、特性列表和误解清单；HiSparse 禁止显式 V1，watermark 覆盖 V1 选择 |
| L1106 | 旧 correctness test 路径删除 | 目标 recursive tree + tests/basic_correctness/models/test_basic_correctness.py L3–5、L125 test_models、L203 test_models_distributed | 改成验证存在的 successor 路径；没有声称运行测试 |

## 不受影响的主干

Executor backend 分派、UniProc 仍拥有 Worker、broadcast/unique reply 区别、Llama load_weights、启动先 distributed 再 snapshot、自动 KV budget、execute/sample 两阶段及 PP 非末级传 intermediates 主干保留。新增功能不等于所有平台/组合都受支持。

## 源码映射同步

在第07章相关引用/实验中替换 tests/basic_correctness/test_basic_correctness.py 为 tests/basic_correctness/models/test_basic_correctness.py。MRV2 sampler 的入口应为 __call__，不是 forward；V1 Sampler.forward 不改。现有 model_runner.py/outputs.py/config 路径仍有效，新增概念的证据就在这些文件。

## 验证边界

2026-10-18 静态源码复核；未安装/运行锁定上游的 GPU runtime，未执行模型加载、CUDA capture/replay、NCCL、D2H trace 或上游 GPU correctness tests。章节保持 draft / runtime_verified: false。

在对应工作副本运行 `python3 -m unittest tests.test_ch07_execution_pipeline tests.test_ch08_async_compile_graph -v`：33 tests passed。两个对应 CPU examples 均退出 0。这些测试只验证随书教学契约，不覆盖本轮新增的上游配置分支。
