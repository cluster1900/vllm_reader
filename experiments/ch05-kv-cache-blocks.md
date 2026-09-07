# 第 05 章实验：KV Cache 容量、共享与回收

本实验分为不依赖 GPU 的确定性部分，以及需要 NVIDIA GPU 和可运行 vLLM 环境的动态部分。
所有性能结果必须记录上游 commit、模型、GPU、启动参数、请求分布和重复次数。

## 1. 基线

```text
source commit: 5893426b88f7b3cd21101d194eb1c6f0a6f0e27b
verified date: 2026-09-07
```

记录环境：

```bash
git -C /Users/hawk_wu/Desktop/vllm rev-parse HEAD
nvidia-smi
python3 --version
```

本仓库当前机器没有可用 vLLM GPU runtime，因此动态部分只给出方法，不伪造结果。

## 2. CPU 教学模型

```bash
cd /Users/hawk_wu/Desktop/vllm_reader
python3 examples/ch05_kv_cache_blocks.py
python3 -m unittest tests.test_ch05_kv_cache_blocks -v
```

预期：10 个测试通过。重点观察：

- block 0 不出现在普通 free queue；
- prefix hit 后相同 block ID 的 ref count 增长；
- cached block 释放后仍在 hash map；
- uncached tail 比 cached prefix 更早复用；
- partial hit 继续写产生 source 到 destination 的 CoW。

## 3. 容量手算

从目标模型配置记录：

```text
num_hidden_layers =
num_key_value_heads =
head_dim = hidden_size / num_attention_heads =
kv_cache_dtype =
tensor_parallel_size =
block_size =
max_model_len =
```

对普通 dense Attention，在 KV heads 能均匀分片的前提下：

```text
local_kv_heads = num_key_value_heads / tensor_parallel_size
bytes_per_token = 2 * layers * local_kv_heads * head_dim * dtype_bytes
bytes_per_block = bytes_per_token * block_size
bytes_per_max_sequence = bytes_per_token * max_model_len
```

若模型使用 MLA、Mamba、Sliding Window、KV 量化、head replication 或 page padding，不要把
该公式当作最终值。改为检查实际 `KVCacheSpec.page_size_bytes` 与 groups。

## 4. 启动日志核对

启动时保存完整日志，查找：

```text
Available KV cache memory
GPU KV cache size
Maximum concurrency
```

建立表格：

| 项目 | 手算 | vLLM 配置/日志 | 差异解释 |
|---|---:|---:|---|
| bytes/token/rank | | | |
| bytes/block | | | |
| available KV bytes | 不适用 | | profile 后结果 |
| num blocks | | | |
| max-length concurrency | | | |

显著差异按顺序检查：

1. 本 rank 实际 layer 数与 PP；
2. KV heads 是否复制或分片；
3. `head_size_v` 是否等于 `head_size`；
4. `tokens_per_state`；
5. `page_size_padded` 与 layout；
6. 多 cache group 的 packed bytes per block；
7. null block 保留；
8. `num_gpu_blocks_override`。

## 5. Prefix sharing 实验

构造两类请求：

```text
shared:  相同的 2048-token prefix + 各自 32-token suffix
control: 不同的 2048-token prefix + 各自 32-token suffix
```

约束：

- 使用 token IDs 验证公共前缀确实相同；
- 模型、sampling、LoRA、multimodal 输入和 cache salt 一致；
- 先单并发预热，再按固定并发重复至少 20 轮；
- 报告中位数和 p90，不只报告最快一次。

记录：

| workload | cached tokens | TTFT p50 | TTFT p90 | throughput |
|---|---:|---:|---:|---:|
| shared | | | | |
| control | | | | |

若 shared 没有命中，检查服务是否启用 prefix caching，以及 prompt tokenization、LoRA、salt
和多模态 extra keys 是否一致。

## 6. Block 压力与抢占

固定 prompt/output 分布，逐步增加并发：

```text
concurrency = 1, 2, 4, 8, 16, ...
```

每一级记录：

| concurrency | KV usage peak | prefix hit | preemptions | TTFT p90 | ITL p90 |
|---:|---:|---:|---:|---:|---:|
| | | | | | |

停止条件：OOM、错误率增加、p99 超过目标或抢占导致吞吐下降。

解释时区分：

- usage 高但没有抢占：可能是健康的容量利用；
- eviction 增加：prefix cache 工作集超过空闲容量；
- preemption 增加：活跃请求的必需 KV 已超出可接纳容量；
- OOM：还要检查非 KV 峰值，不能自动归因于 block pool。

## 7. Block size 对照

只选择当前 backend 支持的 block size。每次只改变 block size，其余配置相同。

记录：

| block size | num blocks | block table length | hit tokens | throughput | TTFT | errors |
|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | |

结论必须同时讨论：

- 尾部内部碎片；
- hash/lookup 粒度；
- block table 和 metadata 数量；
- kernel/backend 约束；
- workload 的 prefix 长度分布。

## 8. Reset 与 eviction 验证

CPU 模型中：

1. 活跃请求存在时调用 reset，应失败；
2. 请求完成后 reset，应成功；
3. cached block 释放后检查 hash 仍存在；
4. 再分配直到该 block 被选中，检查 hash 被移除。

真实 vLLM 中若开启 KV cache events，分别观察 `BlockStored`、`BlockRemoved` 和
`AllBlocksCleared`。不要在有活跃请求时强行清空 cache metadata。
