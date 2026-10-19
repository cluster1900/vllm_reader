# 02-vllm-architecture：v0.32.0 升级复核

- 结论：**已更新（模块迁移、长度语义与输出一致性）**。
- Source: `5df63a7203cdb9328d824a27b682e77c7b9928be`；Reader base: `5f78d105690c1968885abb8c0264cd98fb7b9a18`。
- 日期：2026-10-19；静态源码验证，章节仍为 draft / runtime_verified=false。

## 证据与修订

| 固定源码 | 符号/入口 | 结论与正文修改 |
|---|---|---|
| [vllm/entrypoints/llm.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/entrypoints/llm.py) | `LLM / generate` | 导入 common.offline 并继承 OfflineInferenceMixin，旧文件删除不表示 API 删除。 |
| [vllm/entrypoints/common/offline.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/entrypoints/common/offline.py) | `OfflineInferenceMixin._run_completion / _add_request / _run_engine` | 迁移后的调用链仍含 render、注册、step、收集并按提交序号排序。 |
| [vllm/entrypoints/serve/utils/api_utils.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/entrypoints/serve/utils/api_utils.py) | `get_max_tokens` | 线上取剩余窗口、请求/默认上限、覆盖值、平台上限的最小值，不为完整 max_tokens 预留 prompt 空间。 |
| [vllm/v1/engine/input_processor.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/input_processor.py) | `_validate_prompt_len / process_inputs` | 生成 prompt 等于窗口也拒绝；离线只在 max_tokens 未指定时补剩余窗口。 |
| [vllm/v1/core/sched/utils.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/utils.py) | `check_stop` | 总 token 达到 max_model_len 或输出达到 max_tokens 分别触发长度停止。 |
| [vllm/v1/engine/output_processor.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/output_processor.py) | `process_outputs / _trim_surplus_tokens` | stop string 裁剪一轮多 token 的 surplus，并同步 logprobs、sampling mask 和 routed experts。 |

## 教学示例与验证边界

ch02 模型保留原 max_tokens，同时记录上下文上限，在 Core 计划中取两个停止边界最小值。新增回归分别覆盖超大请求输出/只剩一个位置、输出上限更小、prompt 超长、prompt 恰好满窗口和非正上限。

统一测试结果和逐文件差异覆盖见[总报告](README.md)。历史实验记录不作为此次运行证据。
