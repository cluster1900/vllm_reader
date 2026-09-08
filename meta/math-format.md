# 全书公式写作与构建约定

Markdown 使用支持 LaTeX 数学扩展的预览器；纯 CommonMark 本身不负责数学排版。
本项目统一采用行内 `$...$` 和独立行 `$$` 块公式，支持 GitHub 的数学渲染方式。
[GitHub 官方语法说明](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)。

## 正文中的写法

行内公式用于简短符号和关系，美元符号内侧不留空格；长推导放到独立公式中。
参数名、命令、真实代码和程序输出继续使用反引号，不把它们冒充数学推导。

````markdown
每个候选的分数为 $z_i$。

$$
p_i=\frac{\exp(z_i)}{\sum_j\exp(z_j)}.
$$
````

本项目不再用 math 代码围栏承载正文公式。GitHub 也支持这种围栏，但其他预览器
支持程度不同；统一美元分隔符可以减少两套解析方式造成的差异。

## 排版与语义

公式排版要表达数学含义，不能只让 TeX 解析器接受。

- 函数名用 `\operatorname{softmax}`、`\operatorname{MLP}`，不是斜体字母连乘。
- 描述性下标用 `B_{\mathrm{token}}`；多词说明可用 `\text{...}`。
- 分式用 `\frac{a}{b}`，指数和下标用花括号明确作用范围。
- 单位用正体，如 `128\,\mathrm{KiB/token}`，并说明量纲。
- 多步推导用 `aligned`，矩阵用 `bmatrix`；把等号与相邻内容放在同一行，避免单独
  一行 `=` 被 GFM 当作 Setext 标题标记。
- 舍入值使用 `\approx`。分数向量与单个候选概率、输出 token 数与间隔数必须区分。
- 新引入的变量要就近解释；不要只改公式符号而不更新正文。

## EPUB 的呈现方式

构建先用 Pandoc 解析原生 Markdown，再用锁定版本的 MathJax 将每个 Math 节点转成
自带字形路径的 SVG。公式以图片嵌入 EPUB，不要求阅读器执行 JavaScript、访问 CDN
或安装数学字体；行内公式保留基线位置，块公式按页宽缩放。原始 TeX 保留在替代文本
和 `data-tex` 中，不依赖截图位图或阅读器的 MathML 实现。

SVG 的自包含配置参见 [MathJax 文档](https://docs.mathjax.org/en/v3.2/options/output/svg.html)。
此方案提高阅读器之间的显示一致性，但不能保证任意第三方预览器或阅读器都支持相同
功能。Markdown 预览应启用数学扩展，EPUB 阅读器应支持 EPUB3 与 SVG。

## 检查命令

在仓库根目录运行；Mermaid 使用 `epub/puppeteer-config.json` 配置的现有浏览器，
依赖安装不需要 Puppeteer 再下载浏览器。

```bash
pnpm install --frozen-lockfile
python3 scripts/check_book.py
python3 -m unittest tests.test_math_support
python3 scripts/build_math_preview.py
python3 scripts/build_epub.py --keep-stage
python3 scripts/check_epub.py dist/vllm-source-guide.epub
```

构建会拒绝未知 TeX 命令、缺字、未闭合分隔符及漏解析公式。EPUB 检查逐项对比
公式内容、行内/块模式和出现次数，不再以“至少有一个 MathML 节点”判断全书正确。
`build/epub-src/math-manifest.json` 保存当次公式清单，供显示复核使用。

公式对照预览输出到 `build/math-preview/index.html`，可离线比较原生 Markdown 解析结果
与 EPUB 的 SVG 字形；该预览是排版核对工具，不是新增书籍正文。

## 2026-09-08 验证记录

本轮属于文档排版功能验证，不是模型或 GPU 性能实验。

- Source commit: `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`
- Reader commit: `99c6de62351985e99c69f805a820909efaed5258` 加本轮未提交修改。
- Hardware / OS: Apple Silicon arm64 / macOS 26.6.2。
- Runtime: Python 3.14.3、Node.js 25.2.1、Pandoc 3.9、MathJax 3.2.2。
- Model / vLLM install mode: 不运行模型；输入为全书与术语表的 TeX，vLLM 源码背景仅静态核对。
- Command: 上述检查、预览与 EPUB 构建命令；另以配置的 Chrome 检查预览页和导出的 EPUB XHTML。
- Expected observation: 所有公式可解析，分式/矩阵/下标完整，无漏公式、缺字、损坏图片或源内容不一致。
- Actual observation: 148 个测试通过；170 处公式（89 块、81 行内）与 EPUB 逐项一致；
  22 页桌面公式对照的自动检查均无损坏图片或横向溢出。抽查矩阵、单位、多行等式、
  概率分式、窄屏示例及实际 EPUB XHTML，公式显示正常。
- Conclusion: 已统一 Markdown 公式写法，并通过本轮转换、内容一致性和显示检查。
- Limitations: 未覆盖所有第三方 Markdown 编辑器、EPUB 阅读器、主题与字号组合；
  Markdown 仍需数学扩展，电子书仍需 EPUB3/SVG 支持。
