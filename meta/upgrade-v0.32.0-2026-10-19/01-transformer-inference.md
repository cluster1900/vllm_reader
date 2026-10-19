# 01-transformer-inference：v0.32.0 升级复核

- 结论：**已更新（采样参数和并列候选语义）**。
- Source: `5df63a7203cdb9328d824a27b682e77c7b9928be`；Reader base: `5f78d105690c1968885abb8c0264cd98fb7b9a18`。
- 日期：2026-10-19；静态源码验证，章节仍为 draft / runtime_verified=false。

## 证据与修订

| 固定源码 | 符号/入口 | 结论与正文修改 |
|---|---|---|
| [vllm/sampling_params.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/sampling_params.py) | `SamplingParams.prompt_logprob_token_ids` | 逐 causal row 的二维整数数组/列表，-1 padding 得负无穷；不再是一份共享 ID 列表。 |
| [vllm/v1/engine/input_processor.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/input_processor.py) | `InputProcessor.process_inputs / resolve_watermarking / apply_watermarking` | 实际 prompt 长度确定后校验 rows=max(prompt_len-1-start,0)，请求水印意图与有效配置分开。 |
| [tests/v1/sample/test_topk_topp_sampler.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/tests/v1/sample/test_topk_topp_sampler.py) | `TestTritonTopkTopp` | Triton 恰好 k 项，PyTorch 可多保留第 k 位并列项；测试不再要求两者保留数完全相等。 |
| [vllm/model_executor/models/llama.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/model_executor/models/llama.py) | `LlamaModel / LlamaForCausalLM` | 材料 diff 标为无变化；Transformer、prefill/decode 主干不因本轮参数变化重写。 |

## 教学示例与验证边界

01 两个教学程序的数学路径不涉及逐行 prompt logprob/watermark，也不模拟 Triton backend，代码无影响。正文明确其 top-k 排序截断与生产参考实现的边界。

统一测试结果和逐文件差异覆盖见[总报告](README.md)。历史实验记录不作为此次运行证据。
