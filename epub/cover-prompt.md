# 封面生成记录

- Date: 2026-09-08
- Generation: 内置 `image_gen` 工具；未使用 CLI/API fallback。
- Asset: `epub/cover-imagegen.png`
- 用途：EPUB 封面页及书架缩略图。
- 检查：书名、副标题、作者与 `epub/metadata.yaml` 一致；插画为概念性封面视觉，不作为源码架构图。

## 初始生成提示词（作者信息已由下方编辑更新）

```text
Use case: scientific-educational.
Asset type: finished portrait front cover for a Chinese technical EPUB textbook, flat artwork only, no book mockup or spine.
Primary request: Create an elegant, professionally typeset cover for the existing book titled "vLLM 源码教程", intended for junior and intermediate programmers and software engineering students.
Composition: portrait approximately 2:3, generous margins, very clear typographic hierarchy. Large "vLLM" at top followed by the large exact Chinese title "源码教程". Under the title put the subtitle below in smaller readable type, thoughtfully broken into two or three lines. A sophisticated abstract architectural illustration fills the lower half: ordered token-like tiles travel along precise thin pathways into an organized array of floating layered memory blocks and a simple processor-inspired structure. Convey learning the internals of inference, scheduling and paged memory, without depicting a literal verified architecture diagram.
Style: refined contemporary computer-science book design, crisp geometric forms with gentle depth, restrained technical drawing detail, warm off-white paper ground, deep ink/navy typography with teal and a little muted orange in the illustration. Calm, intelligent, organized and approachable. Title remains legible at small book-library thumbnail size.
Exact text to render, and no other text:
Title: "vLLM" and "源码教程"
Subtitle: "从 Transformer 推理到调度、KV Cache、PagedAttention 与多 GPU"
Author at the bottom: "vLLM Reader Project"
Constraints: accurate Chinese characters and exact Latin spelling, no fake code, no fake formulas, no extra labels, no watermark, no official-company logos, no promotional claims. Final full-bleed cover art with readable typography, not a photograph of a printed book.
```

## 最终作者信息编辑提示词

使用内置 imagegen 编辑上述封面。作者：Hawk Wu；联系邮箱：hawk.wu525@gmail.com。
编辑后图片保存为 `epub/cover-imagegen.png`，书内信息同步到元信息与阅读指南。

```text
Use case: text-localization.
Edit target: the supplied finished Chinese textbook cover.
Change only the bottom author text area. Replace the exact line "vLLM Reader Project" with "Hawk Wu", centered in the same elegant dark navy typography. Add a second, smaller but clearly readable centered line immediately below it containing exactly "hawk.wu525@gmail.com".
Keep generous bottom margins and avoid overlapping the illustration. Preserve the entire existing illustration, title "vLLM 源码教程", subtitle "从 Transformer 推理到调度、KV Cache、PagedAttention 与多 GPU", portrait composition, colors, textures, and all other elements. Do not add any other text or logos. The email must be spelled exactly: hawk.wu525@gmail.com.
```
