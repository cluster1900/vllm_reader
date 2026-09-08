# 阅读指南

本书面向初中级程序员和软件工程类学生。具备 Python 类、函数和循环基础即可开始，
不要求预先掌握大模型推理、CUDA 或分布式系统。全书以 vLLM V1 为主线，从 Transformer 的一次 token 预测开始，逐步进入
EngineCore、Scheduler、KV Cache、PagedAttention、模型执行、CUDA Graph 和多 GPU。

## 作者与联系

- 作者：Hawk Wu
- 联系邮箱：[hawk.wu525@gmail.com](mailto:hawk.wu525@gmail.com)

## 源码基线

- Repository: `vllm-project/vllm`
- Commit: `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`
- Branch: `main`
- Static review: `2026-09-08`
- Runtime GPU verification: not completed

章节中的“源码事实”均绑定上述 revision。标为 `draft` 的章节表示正文已经形成，但真实
NVIDIA GPU 动态实验或独立人工复核尚未全部完成。

## 阅读方法

先按顺序阅读第 01 至 05 章建立请求、调度和内存主线；再阅读第 06 至 09 章理解 GPU
执行和分布式；最后用第 10 章的方法设计实验。代码路径使用仓库相对路径，符号名比
行号更适合在后续版本中重新定位。
