# 第 06 章实验：分页寻址与 Attention Backend

本实验先用纯 Python 验证地址与数值契约，再给出需要 NVIDIA GPU 和可运行 vLLM 环境的
动态核对步骤。不要把教学模型的耗时当作 kernel 性能。

## 1. 基线

```text
source commit: 5893426b88f7b3cd21101d194eb1c6f0a6f0e27b
verified date: 2026-09-07
```

```bash
git -C /Users/hawk_wu/Desktop/vllm rev-parse HEAD
python3 --version
nvidia-smi
```

当前编写环境没有可用的 vLLM NVIDIA GPU runtime，因此 GPU 部分只提供实验方法，不填充
虚构数据。

## 2. CPU 教学模型

```bash
cd /Users/hawk_wu/Desktop/vllm_reader
python3 examples/ch06_paged_addressing.py
python3 -m unittest tests.test_ch06_paged_addressing -v
```

预期：13 个测试通过。重点核对：

- allocation block 32、kernel block 16 时，一个 manager ID 展开为两个 kernel ID；
- 非连续块表 `[2, 0, 3]` 将逻辑位置 `0..9` 映射到
  `[8,9,10,11,0,1,2,3,12,13]`；
- `slot_mapping=-1` 不发生写入；
- 通过块表读取的 attention 与连续逻辑序列参考结果一致；
- 高优先级 backend 被能力条件过滤后，选择下一有效候选。

## 3. 打印真实 backend

在目标 GPU 环境保存完整启动日志，记录：

```text
GPU model and compute capability:
vLLM command:
attention backend:
FlashAttention version if applicable:
KV cache dtype:
framework block size:
kernel block size:
```

源码会记录 `Using ... attention backend`。不要只看安装了哪些包；实际选择还受 head size、
dtype、KV dtype、block size、GPU capability、sliding window、MLA、sinks、非因果模式、
KV connector、PCP/DCP 等条件影响。

## 4. Backend 过滤矩阵

选择一个固定模型，每次只改变一个参数：

| run | dtype | KV dtype | block size | feature | selected backend | rejected reason |
|---|---|---|---:|---|---|---|
| A | bf16 | auto | auto | causal | | |
| B | bf16 | fp8 | auto | causal | | |
| C | bf16 | auto | 16 | sliding window | | |
| D | bf16 | auto | 8 | causal | | |
| E | bf16 | auto | auto | non-causal | | |

若显式指定 backend 后启动失败，这是预期的“严格校验”，不是应该静默回退的情况。自动模式
才会遍历平台候选并选择最高优先级的有效实现。

## 5. Prefill、Decode 与 Mixed Batch

准备三类 workload：

| workload | prompt tokens | generated tokens | concurrency | 目的 |
|---|---:|---:|---:|---|
| prefill-heavy | 8192 | 1 | 1 | 长 query |
| decode-heavy | 128 | 512 | 16 | `query_len=1` 为主 |
| mixed | 混合 | 混合 | 16 | 同轮 ragged query |

每组至少预热 5 次、测量 20 次，记录：

| workload | backend | tokens/step | max query len | max seq len | latency p50/p90 |
|---|---|---:|---:|---:|---:|
| | | | | | |

不要用一次端到端延迟直接断言某个 attention kernel 更快；还需分离 scheduler、模型其他层、
采样、通信和 CUDA Graph 的影响。

## 6. Block Table 与 Slot Mapping 观测

在调试分支中只打印一个小 batch，观察：

```text
request ids
positions
query_start_loc
seq_lens
manager block IDs
kernel block table
slot_mapping
```

手算每个 token：

```text
logical_block = position // kernel_block_size
offset = position % kernel_block_size
physical_block = block_table[request, logical_block]
slot = physical_block * kernel_block_size + offset
```

注意生产日志不要打印完整长序列块表，也不要在热路径执行 `.cpu()` 或 `.tolist()`，这会引入
同步并改变你要测量的行为。

## 7. Profiler 最小实验

使用 PyTorch profiler 或 Nsight Systems 时：

1. 先用固定 workload 确认 backend 和 graph mode；
2. 只捕获少量稳态 step；
3. 标出 KV cache update 与 attention kernel；
4. 区分 prefill、decode、mixed batch；
5. 同时记录 tensor shape 和 metadata；
6. 至少重复三次，报告中位数。

报告模板：

| 项目 | 值 |
|---|---|
| source commit | |
| GPU / driver / CUDA | |
| model / dtype / KV dtype | |
| backend / version | |
| graph mode | |
| batch composition | |
| kernel names | |
| KV update time | |
| attention time | |
| caveats | |

## 8. 失败解释顺序

1. 先看显式 backend 是否与配置兼容；
2. 再看依赖包是否能导入；
3. 再看 GPU compute capability；
4. 再看 dtype、KV dtype、head size 和 block size；
5. 再看 sliding window、MLA、sinks、non-causal 等组合；
6. 最后才怀疑 block table 数值或 kernel 本身。

实验结论必须绑定本节基线。切换 commit 后，重新核对候选优先级、验证条件、metadata 字段和
具体 kernel 调用，不能沿用旧版本结论。
