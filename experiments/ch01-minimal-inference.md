# 第 01 章最小推理实验记录

2026-09-08 全书复核补记：本地 CPU 重跑的版本、环境、命令与结果统一记录在
[全书检查报告](../meta/full-book-review-2026-09-08.md)。下文历史输出保留原日期；GPU
实验表格是待执行模板，空白不代表零值或测试通过。

- Source commit: `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`
- Reader commit: 原始 2026-09-07 记录未保存（当时记录为非 Git worktree）；
  该历史状态无法由本轮复核追溯，不用于描述当前仓库。
- Date: 2026-09-07
- Hardware: Apple Silicon, `arm64`
- OS: macOS 26.6.2, build 25G83
- Python: 3.14.3
- PyTorch / CUDA: 当前解释器未安装 PyTorch，机器未执行 CUDA 路径
- vLLM install mode: 仅静态读取 `/Users/hawk_wu/Desktop/vllm`
- Model: 固定随机权重的单层、单头教学 decoder

## 实验一：KV Cache 等价性

Command:

```bash
python3 examples/ch01_minimal_inference.py
python3 -m unittest tests/test_ch01_minimal_inference.py
```

Expected observation:

- Prefill 与完整 causal forward 的 logits 一致。
- 连续 decode 使用缓存与每步完整重算的最后位置 logits 一致。
- `temperature=0` 和 `top_k=1` 都选择最大 logit。

Actual observation:

```text
prompt: <bos> 我 喜欢 读
prefill cache entries per layer: 4
decode step 1: max cached/full diff=0.00e+00, sampled=5:学习
decode step 2: max cached/full diff=0.00e+00, sampled=5:学习
decode step 3: max cached/full diff=0.00e+00, sampled=6:。
QK score count example (P=128, G=32): no-cache=333136, cached=12720

Ran 4 tests in 0.002s
OK
```

Conclusion:

在该教学实现中，缓存历史 K/V 不改变计算结果，并显著减少生成阶段重复的 QK score
计算。这个实验验证数学结构，不证明真实 vLLM 的端到端加速比例。

## 实验二：Temperature 与手算 Attention

Command:

```bash
python3 examples/ch01_numerical_walkthrough.py
python3 -m unittest tests/test_ch01_numerical_walkthrough.py
```

Expected observation:

- 相同 logits 下，temperature 增大时最高概率下降、分布变平。
- attention probabilities 总和为 1，输出等于 Value 的加权和。

Actual observation:

```text
T=0.5: [0.9793, 0.0179, 0.0024, 0.0003]
T=1.0: [0.8310, 0.1125, 0.0414, 0.0152]
T=2.0: [0.5793, 0.2131, 0.1293, 0.0784]
attention scores: [0.7071, 0.3536, 1.0607]
attention probabilities: [0.3199, 0.2246, 0.4555]
attention output: [1.2309, 0.9047]
```

Conclusion:

Temperature 只改变采样分布，不改变模型已经计算出的原始 logits。Attention 的输出
是按 Query/Key 匹配权重对 Value 做加权聚合。

## 限制与待运行项

- 随机权重不具备语言能力。
- Python list 实现不反映 GPU kernel、显存带宽和 launch overhead。
- 尚未使用真实 tokenizer 和预训练模型验证逐 token 文本输出。
- 尚未在 NVIDIA GPU 上 trace `GPUModelRunner -> LlamaForCausalLM -> Attention -> Sampler`。
- 第 01 章只有在上述 runtime trace 完成并复核后，才可从 `draft` 改为 `verified`。
