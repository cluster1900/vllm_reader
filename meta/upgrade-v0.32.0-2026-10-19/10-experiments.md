# 10-experiments：v0.32.0 升级复核

- 结论：**已更新（sweep、profile 与缓存指标）**。
- Source: `5df63a7203cdb9328d824a27b682e77c7b9928be`；Reader base: `5f78d105690c1968885abb8c0264cd98fb7b9a18`。
- 日期：2026-10-19；静态源码验证，章节仍为 draft / runtime_verified=false。

## 证据与修订

| 固定源码 | 符号/入口 | 结论与正文修改 |
|---|---|---|
| [vllm/benchmarks/sweep/serve.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/benchmarks/sweep/serve.py) | `run_comb` | warmup.json 独立、warmup run_number=-1，不进入正式测量；continue_on_error 记录 failure 并返回 None。 |
| [tests/benchmarks/sweep/test_param_sweep.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/tests/benchmarks/sweep/test_param_sweep.py) | `test_run_comb_excludes_warmup_from_measured_results / test_run_comb_continue_on_error_keeps_later_runs` | 上游测试证据：warmup 与计量分离、失败不阻断后续运行。这里只读测试，不声称执行上游 pytest。 |
| [docs/contributing/profiling.md](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/docs/contributing/profiling.md) | `Profile with PyTorch Profiler` | session prefix、delay/max iterations 与 409 会话边界；Worker 迭代上限不限制 AsyncLLM CPU trace，仍需 stop。 |
| [vllm/config/profiler.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/profiler.py) | `validate_profile_iteration_bounds` | 非负整数校验，不把字符串或浮点悄悄转为整数。 |
| [vllm/v1/metrics/loggers.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/metrics/loggers.py) | `PrometheusStatLogger` | KV consumer 才注册 fetch stage gauge；cache source counter 区分 device/host/disk/p2p/external。 |

## 教学示例与验证边界

ch10 教学程序只计算已有样本的 TTFT/ITL、吞吐、置信区间与 Pareto，不模拟 sweep 执行器或 profile session，因此代码/测试无语义变更；本次仍实际重跑。未运行任何上游服务 benchmark 或性能实验。

统一测试结果和逐文件差异覆盖见[总报告](README.md)。历史实验记录不作为此次运行证据。
