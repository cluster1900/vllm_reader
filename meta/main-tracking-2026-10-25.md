# vLLM main 跟踪记录（2026-10-19～2026-10-25，未发版）

- 当前阅读基线：`v0.32.0` / `5df63a7203cdb9328d824a27b682e77c7b9928be`（基线 003），正文不变。
- 跟踪上游：`vllm-project/vllm` `main`，截至 `a7407ab44`（[#63616](https://github.com/vllm-project/vllm/pull/63616)，2026-10-25）。
- 方法：在只读 clone 中逐条 `git show` 阅读下列提交的源码 diff；只记录读过 diff 的条目。没有运行上游代码、GPU 或模型。
- 结论：本窗口内没有发现 v0.32.0 正文对 v0.32.0 本身描述错误的段落；与 main 行为不同处在章节内加“main 上的变化（未发版）”注记。

## 影响表

| SHA / PR | 领域 | main 上的变化 | 本书位置 | 处理 |
|---|---|---|---|---|
| [ceb01b6dd](https://github.com/vllm-project/vllm/commit/ceb01b6ddf6e19510ba138f810d38dd70e665f52) / [#61976](https://github.com/vllm-project/vllm/pull/61976) | 调度抢占 | PRIORITY 下抢占对象是自身时 `continue` 而非 `break` | ch04 §4.8 末 | 注记 |
| [43caf5b35](https://github.com/vllm-project/vllm/commit/43caf5b3520b704fa766f00fc2b24da25c78e084) / [#59030](https://github.com/vllm-project/vllm/pull/59030) | 调度准入 | encoder/lookahead 后 0 token 由 `break` 改为移入 step-skipped 队列并 `continue`；关闭 chunking 超预算、padding、Mamba 对齐的 `break` 不变 | ch04 §4.9.1 | 注记 |
| [d5b36699b](https://github.com/vllm-project/vllm/commit/d5b36699b21bd7b1c4ce01afe218e3c3eaacc27e) / [#51488](https://github.com/vllm-project/vllm/pull/51488) | EngineCore 暂停 | `abort`/`wait` 暂停期间新请求返回可重试 `engine_paused` 错误；`keep` 不变 | ch03 §3.12 模式表“新请求”列与暂停状态图 | 注记 |
| [636da2476](https://github.com/vllm-project/vllm/commit/636da2476d184b44e670449498ec850fbfaab8d6) / [#60513](https://github.com/vllm-project/vllm/pull/60513) | spec decode（MRV2） | draft speculator 在 auto-fit 后读取 `max_model_len`，`max_seq_len` 取较小值 | ch03 auto-fit | 注记 |
| [7050ef881](https://github.com/vllm-project/vllm/commit/7050ef881ff0835f6890a47db1859a24c65481d6) / [#58466](https://github.com/vllm-project/vllm/pull/58466) | 采样 / spec decode | 允许 batch-sharded sampling 与 adaptive verification 同时开启 | ch09 batch-shard 段 | 注记 |
| [5f63ea461](https://github.com/vllm-project/vllm/commit/5f63ea461ccd567ece6f50dd3701430c0f321039) / [#63001](https://github.com/vllm-project/vllm/pull/63001) | spec decode | 支持加载 ModelOpt 混合精度 DFlash/DSpark draft | ch09 DSpark 段 | 仅清单（新增能力，不冲突） |
| [fc2b6ab45](https://github.com/vllm-project/vllm/commit/fc2b6ab45ed3e498b52d74fd880a8b40a65ba3a7) / [#62698](https://github.com/vllm-project/vllm/pull/62698) | MRV2 staged write | apply-write kernel 改为 (行, 1024 元素 chunk) 二维 grid | ch07 §7.12.1 | 注记 |
| [2c19f6793](https://github.com/vllm-project/vllm/commit/2c19f6793e98384ee38ba9b5df425b7378f3ae95) / [#63372](https://github.com/vllm-project/vllm/pull/63372) | MRV2 / DBO | routed-experts capturer 按 ubatch token 偏移写入 | ch08 DBO 段 | 注记 |
| [66bf93154](https://github.com/vllm-project/vllm/commit/66bf931544202f159cf83c66a812826b7d5c84fb) / [#63476](https://github.com/vllm-project/vllm/pull/63476) | MRV2 / DBO | 报告最先记录的失败微批 | ch08 DBO 段 | 注记 |
| [1757f6720](https://github.com/vllm-project/vllm/commit/1757f6720c3c1cc130a834b41f264ac22a782289) / [#61390](https://github.com/vllm-project/vllm/pull/61390) | KV offload | fs 发布 EEXIST 视为已存储 | ch05 §5.15.4 | 注记（汇总） |
| [6f5143383](https://github.com/vllm-project/vllm/commit/6f51433838b858a9cd0478b2ffee4d2c610a1712) / [#63010](https://github.com/vllm-project/vllm/pull/63010) | KV offload | 重启时重置 next_stored_chunk_idx | ch05 §5.15.4 | 注记（汇总） |
| [2b1f75c80](https://github.com/vllm-project/vllm/commit/2b1f75c80f79496b313e28362f0856632dfbf5ea) / [#61780](https://github.com/vllm-project/vllm/pull/61780) | KV offload | store 失败时调用背压检测 on_store_failed；仅 fs/obj 存储层设阈值，连续 3 次失败进入背压（p2p 等不适用） | ch05 §5.15.4 | 注记（汇总） |
| [921ed7fa1](https://github.com/vllm-project/vllm/commit/921ed7fa11c41b89ce36e81f50fb3b1c293dc8f7) / [#60810](https://github.com/vllm-project/vllm/pull/60810) | KV offload | 恢复 session 时 is_full_table 重置 block table | ch05 §5.15.4 | 注记（汇总） |
| [fc0cfcdc0](https://github.com/vllm-project/vllm/commit/fc0cfcdc05b1717cadb17a0e24a6b1f611aec456) / [#61019](https://github.com/vllm-project/vllm/pull/61019) | KV offload | EAGLE/MTP group 的 partial tail | ch05 §5.15.4 | 注记（汇总） |
| [65bc69835](https://github.com/vllm-project/vllm/commit/65bc698356b4b0106630788536b6af6a916308e9) / [#63266](https://github.com/vllm-project/vllm/pull/63266) | KV offload | 拒绝未对齐的 ChunkedLocal CPU 后缀命中 | ch05 §5.15.4 | 注记（汇总） |
| [4e5916571](https://github.com/vllm-project/vllm/commit/4e591657157a021cab997f911d67f50c0d18a71e) / [#63423](https://github.com/vllm-project/vllm/pull/63423) | 结构化输出 | lm-format-enforcer 后端弃用（告警），计划 v0.35.0 移除；改用 auto/xgrammar/guidance | 无章节提及 | 仅清单 |
| [e7473bf68](https://github.com/vllm-project/vllm/commit/e7473bf6862289f0647d1f4b01498288b181e063) / [#63466](https://github.com/vllm-project/vllm/pull/63466) | 依赖 | flashinfer-python 0.7.2 → 0.7.2.post1 | 书中无版本号 | 仅清单 |
| [d4d647b2b](https://github.com/vllm-project/vllm/commit/d4d647b2b52d5e15ee1427de0effecd474430972) / [#63253](https://github.com/vllm-project/vllm/pull/63253) | 依赖 | DeepGEMM 固定 tag 更新 | 书中无版本号 | 仅清单 |

## 升级清单（下一个 tag 发版后）

- [ ] 确认新 tag 是否包含上表各提交（release 分支可能不含 main 全部提交）。
- [ ] ch04：按 #61976、#59030 改写 §4.8 / §4.9.1 的 break/continue 描述，删除注记。
- [ ] ch03：按 #51488 更新暂停状态图与模式表；按 #60513 更新 auto-fit 段。
- [ ] ch09：按 #58466 更新 batch-shard 限制；评估是否补充 #63001。
- [ ] ch07 / ch08：按 #62698、#63372、#63476 更新 staged write 与 DBO 段。
- [ ] ch05：复核 KV offload 修复是否需要进入正文。
- [ ] 若书中加入结构化输出后端列表，标注 lm-format-enforcer 已弃用（#63423）。
- [ ] 更新 chapter-source-map.json 的 source_commit，跑 check_book.py 并新增阅读基线 004。
