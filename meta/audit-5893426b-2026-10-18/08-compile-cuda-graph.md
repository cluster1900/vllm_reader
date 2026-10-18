# 第08章：旧基线审阅

Reader 原文：7d4cd098cf6ec1abd8c7b570dc1df5238cf37eef；源码基线：5893426b88f7b3cd21101d194eb1c6f0a6f0e27b。原文行号相对于 book/08-compile-cuda-graph/README.md。

| 原文位置 | 问题 | 源码证据（旧基线） | 修正 |
|---|---|---|---|
| L1421–1437 / §8.15.2 | 将 mixed FULL + NEVER 写成自动降级，实际直接报错；图的 mixed 降级分支不继续 decode/spec 检查 | vllm/config/compilation.py L1395–1425：NEVER 的 ValueError；L1428–1465：后续独立 decode/spec 检查 | 分离 fatal 分支，mixed 降级回到后续检查；重写图与图注，使顺序与代码一致 |

## 覆盖与保留结论

通读全部 8.1–8.25、图示/图注和实验教学边界。复核 EngineCore.step/batch queue、AsyncScheduler placeholder/stale 更新、AsyncOutput copy event、compile modes/decorator/piecewise backend、MRV1 dispatcher 与 MRV2 descriptor/capture/output buffers、graph profiling。原先 grammar 延迟采样例外、默认 graph 预留与 GPU 完成事件不等于计算进度的修订继续保留。数值 28/22ms 是教学时序，不标为设备性能。

## 验证边界

2026-10-18 静态源码复核；未安装/运行锁定上游的 GPU runtime，未执行模型加载、CUDA capture/replay、NCCL、D2H trace 或上游 GPU correctness tests。章节保持 draft / runtime_verified: false。

在对应工作副本运行 `python3 -m unittest tests.test_ch07_execution_pipeline tests.test_ch08_async_compile_graph -v`：33 tests passed。两个对应 CPU examples 均退出 0。这些测试只验证随书教学契约，不覆盖本轮新增的上游配置分支。
