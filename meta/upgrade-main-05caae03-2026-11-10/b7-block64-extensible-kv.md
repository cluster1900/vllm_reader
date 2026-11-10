# 第 05 章：默认 block size 64 与 Extensible KV（基线 `05caae03`）

- 读者仓库基线（本说明初稿）：`cluster1900/vllm_reader` `efaf5042305ee28f2bb585f2eb1ce765bafe9e0f`（#15 合入后的 `main`）。r2 已用普通 merge 同步到 `96afa21df2c2847488d153f99c244cd4908c2cee`（#17）。
- 上游 pin：`05caae03c20f69436ff7ec23ded339ea104f1a01`。跟踪：`cluster1900/bot-hub#127`。
- 章节仍为 draft，`runtime_verified=false`；全部静态核对，没有跑 GPU。
- 本批只改第 05 章正文、`meta/chapter-source-map.*` 的第 05 章条目，以及本说明。未改第 07 章交叉引用，留给后续。

## 一、默认 `--block-size` 16 → 64

| 项 | 值 |
|---|---|
| 提交 | `732e49d4420448a88dbdbbdb80a765bf51981e51`（#63231） |
| 祖先 | 是 `05caae03` 的祖先；**不是** v0.33.0 `4ba787ec` 的祖先 |
| 源码 | `vllm/config/cache.py:185` `CacheConfig.DEFAULT_BLOCK_SIZE = 64`；未指定时由 `_apply_block_size_default`（`:475`）写入 |

正文改动：

- §5.2.1 教学表、每块公式和随书 assert 改为配置默认 64 token / 8 MiB；每个 token 仍是 128 KiB，8192-token 序列仍是 1 GiB。旧默认 16 只作为历史提一次。正文写明 64 只是 `CacheConfig` 配置默认，启动时平台、backend 和混合模型对齐可能改成别的值（`Platform.update_block_size_for_backend`，`interface.py:760`）。显式传入 16 仍会跳过 backend 偏好改写，但混合对齐和部分平台仍可能覆盖。
- §5.16.1 的 `T=16` 是为了整除好看的教学取值，不是当前配置默认，予以保留并注明；同时写明启动时最终块大小仍可能与配置默认不同。

未改：kernel「16 的倍数」、manager→kernel 拆页算例（第 06 章），以及教学模型单元测试里显式传入 16 的断言（该测试检验的是辅助函数，不是默认值）。

## 二、Extensible KV cache

| 项 | 值 |
|---|---|
| 主提交 | `860da3dafdbb1e24a6aef821b4ca7bceeb9b87d8`（#56492） |
| 祖先 | 是 `05caae03` 的祖先；**不是** v0.33.0 的祖先 |
| 正文位置 | 新 §5.6.5，只写生命周期：enable → resolve → 预留 → finalize → extend |

关键符号（行号均在 `05caae03` 上核对）：

- `CacheConfig.enable_extensible_kv_cache`（`cache.py:388`）
- `VllmConfig._resolve_extensible_kv_cache`（`vllm.py:3176`）
- `ExtensibleKVCache` / `extend_kv_cache`（`extensible_kv_cache.py:71`、`:450`）
- `finalize_extensible_kv_cache`（`kv_cache_planning.py:1839`）
- `EngineCore._initialize_kv_caches` 中的 extensible 分支（`core.py:312-317` 第二道门、`:413` finalize、`:422` extend）
- `gpu_worker.py:1233-1243` warmup 结束后量测；`ensure_kv_cache_blocks`（`extensible_kv_cache.py:443`）只在 warmup 期间按需 commit

后续相关提交（行为边界已在 §5.6.5 一句带过，未展开实现）：#65225（segment padding）、#65463（`--max-model-len auto` 再拟合）、#65413 / #63401（UMA / ROCm 显存量测）、#65137 / #65331 / #65214（NIXL / Mooncake / HiSparse 适配）。

§5.6.1 第 3 项已有「开启 extensible 时 `profile_cudagraph_memory()` 返回 0」，本批补了指向 §5.6.5 的交叉引用。

## 三、有意未改

- 第 07 章 §7.9 / §7.10 的 Extensible KV 交叉引用：不在本批文件范围内，列入 PR 说明，留给后续。
- 本批不改 `meta/upgrade-main-05caae03-2026-11-10/b6-mrv1-rebind.md` 与 `meta/source-revisions.md`（前者由 #16 维护）。#56492 与 block_size=64 现已写入第 05 章正文（§5.2.1、§5.6.5），并因此取代 `b6-mrv1-rebind.md:91`（「#56492 extensible KV cache」仍列在「本批没有写入正文的上游漂移」）与 `source-revisions.md:114`（「block_size=64、extensible KV … 不在本批，… 未写入正文」）里的表述；那两处文件本身不动，留给其所有者后续更新。
- 第 05 章正文里原先没有把 #56492 写进「正文尚未展开」注记，故无需从该注记删除。

## 四、图与公式

- §5.2.1 容量公式保留，数字改为默认 64，并补白话说明。
- §5.6.5 新增生命周期流程图，附读图方法。r2 将顺序改为 finalize 后 extend（`FIN → EXT`），已用 mermaid-cli 重新渲染并目视检查。
