# 第07章：旧基线审阅

Reader 原文：7d4cd098cf6ec1abd8c7b570dc1df5238cf37eef；源码基线：5893426b88f7b3cd21101d194eb1c6f0a6f0e27b。以下行号为 original book/07-model-execution/README.md。

| 原文位置 | 问题 | 源码证据（旧基线） | 修正 |
|---|---|---|---|
| L161 | MRV2 引用不存在的 Sampler.forward，混入 MRV1 API | vllm/v1/worker/gpu/sample/sampler.py L122–205：Sampler.__call__；vllm/v1/sample/sampler.py L73 是另一套 Sampler.forward | 改为 __call__，保留 MRV1 forward 引用 |
| L768–780 | 状态图把再次 execute 统一画作 Error，易误认为 MRV2 有同一防御检查 | vllm/v1/worker/worker_base.py L165–175 两阶段契约；gpu_model_runner.py L4279–4283 明确 RuntimeError；gpu/model_runner.py execute_model 至 L1884 赋 state 无同等入口检查 | 图改 ProtocolViolation，说明是契约违规；实际显式报错限定 MRV1 |
| L934 | 输出表把 logprobs 简称概率，混淆 logits/log probability/probability | vllm/v1/sample/sampler.py L73 起 raw/processed logits/logprobs；gpu/sample/sampler.py L122 起 get_logprobs | 明确配置相关对数概率或 logits，不是归一化概率 |

## 覆盖与保留结论

通读全章 7.1–7.31、首尾教学与实验段落及图注；重点沿 Executor.get_class → UniProc/Multiproc → Worker.init_device/load/profile → MRV1/MRV2 execute/sample → AsyncOutput → Scheduler 回查。
保留已正确修订的 graph 估算默认 1、通信初始化先于内存快照、单输出 rank 不等于单计算 rank、增量状态/永久行、grammar 汇合点。GPU memory profiling 的 opt-in 注释与默认值冲突已在原文清楚披露，不重复“修复”。旧基线不采纳新版本 HiSparse/watermark/aux-output 行为。

## 验证边界

2026-10-18 静态源码复核；未安装/运行锁定上游的 GPU runtime，未执行模型加载、CUDA capture/replay、NCCL、D2H trace 或上游 GPU correctness tests。章节保持 draft / runtime_verified: false。

在对应工作副本运行 `python3 -m unittest tests.test_ch07_execution_pipeline tests.test_ch08_async_compile_graph -v`：33 tests passed。两个对应 CPU examples 均退出 0。这些测试只验证随书教学契约，不覆盖本轮新增的上游配置分支。
