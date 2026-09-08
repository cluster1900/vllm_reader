# EPUB 封面与 Markdown 一致性检查

本轮已用内置 imagegen 生成新封面，并将其嵌入 EPUB。按照阅读内容逐项比较，
阅读指南、10 章正文和术语表均与当前 Markdown 一致。

## 封面

- 图片：[cover-imagegen.png](../epub/cover-imagegen.png)
- 生成方式与完整提示词：[封面生成记录](../epub/cover-prompt.md)
- 书名、副标题、作者与 `epub/metadata.yaml` 一致，已目视复核。
- EPUB3 的 `cover-image`、兼容阅读器的 `meta name="cover"` 和第一张封面页引用同一图片。
- 包内封面与选定 PNG 的字节内容相同；已从实际 EPUB 展示封面页和扉页进行显示检查。

## 正文来源与比较规则

正文按以下顺序构建：

1. `book/reading-guide.md`
2. `book/01-transformer-inference/README.md` 至 `book/10-experiments/README.md`
3. `glossary/terms.md`

仓库首页、写作路线图和协作规范属于项目文档；实验与教学代码通过保留原文字的链接
指向仓库固定提交，不作为额外正文插入 EPUB。

比较使用独立的 Markdown → HTML 解析结果，对照 EPUB 阅读顺序中的实际 XHTML：

- 正文文字、标题、段落、列表、表格单元格、强调、折叠答案和代码逐项比较。
- 保留代码缩进；忽略普通正文换行、排版包装节点及代码末尾的换行差异。
- 公式按 TeX 内容、行内/块模式、出现次数和所在顺序比较，并核对对应 SVG。
- Mermaid 原文编码保存在 SVG 中，并用摘要与 Markdown 图源逐张核对；普通插图检查文件内容。
- 目录按标题文字、顺序和跳转目标比较；包内链接检查目标文件及锚点。
- 每份正文的原始 Markdown 摘要写入章节标记，包含 YAML 元信息的变化也会触发重建检查。
  YAML 不作为正文直接排印，版式与文件编码的差异不属于内容差异。

## 修复的导出问题

旧构建会给仓库链接额外附加“随书仓库”说明，使导出文字与 Markdown 不同；本轮改为
只转换链接目标，保留链接文字和格式。阅读指南从脚本内字符串迁至 Markdown，避免
两份内容独立维护。封面从字体依赖的 SVG 改为 imagegen PNG，并补齐兼容封面标记。

## 封面与作者阶段的验证记录

- Source commit: `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`，上游只读。
- Reader build base: `48c6e74e1216180cb9a49267c4f5ffaf5c98c631` 加本轮工作区修改。
- Date: 2026-09-08。
- 环境：macOS / arm64，Python 3.14、Pandoc 3.9；使用项目配置的 Chrome 做显示检查。
- `python3 -m unittest discover -s tests`：159 个测试通过。
- `python3 scripts/check_book.py`：10 章、172 个主源码锚点、538 处引用检查通过。
- `python3 scripts/check_epub.py dist/vllm-source-guide.epub`：12 份正文来源、56,288 条有序
  结构化检查记录全部匹配；170 处公式、294 张 Mermaid 图、目录、锚点与封面均通过。
- 额外反证：在临时 EPUB 副本中插入一段文字，保持标题、公式及图片数量不变，校验器
  仍准确拒绝该副本并报告发生差异的章节和位置。交付文件没有这段测试文字。

可重复构建与复核：

```bash
python3 scripts/build_epub.py --keep-stage
python3 scripts/check_epub.py dist/vllm-source-guide.epub
```

交付文件为 `dist/vllm-source-guide.epub`。本轮验证针对发布内容和导出一致性，
不将其表述为 vLLM GPU 运行验证，也不保证所有第三方阅读器使用完全相同的字体与换行。


## 作者信息补充

作者已更新为 **Hawk Wu**，联系邮箱为 **hawk.wu525@gmail.com**。封面使用内置 imagegen
编辑，扉页由 `epub/metadata.yaml` 与 `epub/template.xhtml` 生成，阅读指南保留相同
作者与可点击的邮箱链接。已检查实际 EPUB 的 `dc:creator`、扉页作者和 `mailto:` 地址，
并重新通过全书一致性校验。

## 后续全书复核

随后进行的源码与教学复核见 [全书再次复核](final-source-review-2026-09-08.md)。
上面的数量记录属于封面制作阶段；后续正文修订会改变有序检查记录数量，
最终 EPUB 随当前 Markdown 重建，并重新执行相同的完整内容校验。
