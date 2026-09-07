#!/usr/bin/env python3
"""Add consistent reader guidance to every chapter and Mermaid diagram."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book"
CHAPTER_GUIDES = {
    "01-transformer-inference": """## 如何阅读本章

你可以把这一章当作“使用过 LLM 之后的原理补课”。我们默认你知道 prompt、token、
temperature 和上下文长度这些产品概念，但不要求你学过深度学习课程。第一次阅读只需
回答三个问题：数据长什么样、一次 forward 做什么、为什么生成必须重复 forward。

遇到向量和矩阵公式时，先看 shape，不要先做代数推导。例如 `[T, D] -> [T, V]`
首先表示“每个待计算 token 的隐藏向量被变成一组词表分数”。公式中的求和、softmax
和旋转随后只是解释这些 shape 内部怎样计算。书中的小数字例子用于建立直觉，不要求
你手算真实模型的 4096 维向量。

本章可以使用一条贯穿始终的类比：token ID 是字典编号，embedding 是查到的工作向量，
attention 是按相关性翻阅历史笔记，KV Cache 是保存已经整理好的历史笔记，sampler
则根据模型给出的候选分数做最后选择。类比只帮助入门，进入源码时仍以张量、位置和
缓存所有权为准。
""",
    "02-vllm-architecture": """## 如何阅读本章

这一章最容易出现的问题不是数学太难，而是类名太多。不要尝试一次记住所有对象；先把
一次请求当成一张工单，观察它如何经过“接单、标准化、排队、执行、交付”五个阶段。
之后再把 Renderer、EngineCore、Scheduler、Worker 和 OutputProcessor 放回对应阶段。

第一次阅读只追一条普通文本生成的 happy path，看到多模态、structured output、DP
或异常分支时先知道它挂在哪里即可。第二次阅读再区分函数调用、线程队列、ZMQ 和 GPU
边界，因为同一个 Python 方法名跨过这些边界后，性能和失败传播方式会完全不同。

本章不要求你理解 attention kernel。你需要建立的是系统地图：谁拥有请求状态，谁拥有
调度权，谁拥有设备内存，谁把 token 还原成用户看到的文本。后续章节都是对这张地图的
局部放大。
""",
    "03-engine-core": """## 如何阅读本章

可以把 EngineCore 看成推理系统的主控循环，而不是“另一个模型类”。它反复做三件事：
接收新请求，向 Scheduler 询问本轮工作，把工作交给 Executor 并结算结果。先掌握这个
循环，再研究同步、异步和多进程版本如何改变等待方式。

阅读状态机和队列代码时，建议在纸上保留三栏：新输入、在途 batch、已返回结果。每看到
一次 enqueue、dequeue、schedule 或 update，就把对象在三栏之间移动。这样比记忆每个
成员变量更容易发现背压、乱序结果和 shutdown 的真实含义。

本章涉及的“异步”主要是控制流并发，不等于 GPU kernel 自动并行。看到 async、线程或
ZMQ 时先问“谁可以继续做别的工作”，再问“结果由谁、在何时确认”。
""",
    "04-scheduler": """## 如何阅读本章

Scheduler 可以先理解为一个受多种预算约束的工单调度员：请求很多，但每轮只有有限的
token 预算、序列名额和 KV block。它不是预测模型，也不是简单地从队列头取固定数量的
请求，而是在每个 step 重新分配稀缺资源。

本章公式优先按“账本”来读。`num_computed_tokens` 是已经完成的工作，目标 token 数是
当前希望推进到的位置，两者之差就是待安排工作。先用几个整数手算，再看 chunked
prefill、speculative decoding 和 async scheduling 如何复用同一套差额模型。

第一次阅读抓住 waiting、running、preempted 三种状态和一次 `schedule()` 的主路径；
第二次再看 priority、encoder budget、remote KV 和 fence。调参前必须先明确优化的是
TTFT、TPOT、吞吐还是公平性，因为同一个参数不可能同时把所有指标都推向更好。
""",
    "05-kv-cache": """## 如何阅读本章

可以把 KV Cache 管理看成仓库系统：GPU 上的 K/V tensor 是仓库建筑，block 是统一编号
的货位，block table 是每个请求的货位清单，prefix cache 则让多个请求复用已经装好的
货位。这个类比能解释分配和回收，但物理 tensor、元数据对象和逻辑 token block 仍要
严格区分。

涉及容量公式时先做单位检查。把层数、head 数、head size、dtype 字节数和 token 数逐项
写出单位，最后结果才应是 byte。只要单位对不上，就说明把 query head、KV head、block
大小或并行切分混在了一起。

第一次阅读只追一个请求从申请 block 到释放 block；第二次加入共享前缀和引用计数；
第三次再看多 cache group、KV connector 和平台差异。每一步都问“谁拥有这块内存”和
“谁只持有编号”，这是避免混淆的关键。
""",
    "06-paged-attention": """## 如何阅读本章

PagedAttention 最适合借用操作系统虚拟内存的直觉：请求看到的是连续 token 位置，实际
K/V 可以散落在不同物理 block；block table 负责把逻辑位置翻译成物理位置。它不是说
attention 数学变了，而是 attention 读取历史 K/V 的寻址方式变了。

阅读地址公式时，把除法和取模分别翻译成两个问题：这个 token 属于哪个 block，以及它
位于 block 内第几个 slot。再把逻辑 block ID 通过 block table 换成物理 block ID，就能
理解 slot mapping，而不需要先掌握 CUDA thread block 的细节。

第一次阅读停在“元数据如何到达 backend”；第二次再进入 prefill/decode kernel 路径；
第三次才比较 FlashAttention、FlashInfer 等 backend。backend 名称会变，但逻辑位置、
物理位置和元数据契约这三层长期更稳定。
""",
    "07-model-execution": """## 如何阅读本章

执行栈可以类比为一条生产线：Executor 决定调用哪些工作站，Worker 管理一个设备环境，
Model Runner 把调度结果装配成 tensor，模型执行数值计算，Sampler 选出 token。名字都
带“执行”，但它们拥有的资源和状态完全不同。

阅读本章时先给每个对象贴三个标签：运行在哪个进程、拥有哪个设备、输入输出是什么。
只要这三个问题能回答，就不会把 Executor 的分布式编排、Worker 的设备生命周期和
Runner 的 batch 构造混在一起。

第一次阅读追单 GPU、普通文本生成的完整 step；第二次加入 TP/PP 广播和返回值规则；
第三次再看 MRV1/MRV2、warmup、CUDA Graph 与异步输出。模型内部算子已经在第 01 和
第 06 章建立基础，本章重点是它们怎样被系统正确地调用。
""",
    "08-compile-cuda-graph": """## 如何阅读本章

这一章包含三个容易混淆但彼此独立的问题：CPU 与 GPU 工作能否重叠、PyTorch 图能否
编译优化、CUDA kernel 启动序列能否捕获后 replay。先分别理解三者解决的开销，再看
vLLM 如何把它们组合起来。

不需要预先掌握编译器理论。遇到 graph、guard、dynamic shape 和 capture size 时，先
用“录制一条固定条件下可重复执行的流水线”理解：条件变化可能重新编译，地址变化可能
无法 replay，shape 不匹配则需要 padding、换图或回退 eager。

第一次阅读只比较 eager、compile、CUDA Graph 三条路径；第二次研究 async scheduler
的 placeholder 和确认边界；第三次再进入 MRV1/MRV2 dispatcher。性能图必须同时看
冷启动、稳定态、显存和正确性，不能只比较一个平均延迟数字。
""",
    "09-distributed": """## 如何阅读本章

学习分布式时不要从缩写表开始。先问“什么东西太大或太慢”，再决定切权重、切层、切
请求、切专家还是切上下文。每种并行方式都可以用三问描述：切分对象是什么，哪些 rank
必须通信，通信结果在哪里重新合并。

rank 公式优先画坐标而不是死记乘法。把一个 worker 看成位于 DP、PP、PCP、TP 等轴上的
一个坐标点，process group 就是在固定其他坐标后沿某一轴取出的一组点。DCP 复用 TP
worker，因此不会像新轴一样增加进程数，这是本章尤其需要避免的误区。

第一次阅读只掌握 TP、PP、DP；第二次加入 EP、PCP、DCP；第三次再看 executor、NCCL
和跨节点部署。任何“并行更快”的结论都必须同时核算计算量、通信量、显存和请求规模。
""",
    "10-experiments": """## 如何阅读本章

这一章不是命令清单，而是一套证明优化有效的方法。先写假设，再选指标、控制变量和负载，
最后才运行 benchmark。没有假设的 profile 很容易得到大量时间线，却不知道下一步该改
什么。

指标公式不需要复杂统计背景，但必须分清分子、分母和观察者。TTFT、TPOT、ITL、E2EL
描述的时间区间不同；吞吐和 goodput 对“完成”也有不同要求。先在单个请求时间线上标点，
再聚合百分位，能避免把客户端排队、服务端排队和 GPU 执行混成一个数字。

第一次阅读建立指标词典和五层 benchmark；第二次学习 profiler 与诊断树；第三次设计
完整实验矩阵。任何结果至少同时报告正确性、延迟分布、吞吐、资源使用和实验环境，单个
峰值数字不能支持生产结论。
""",
}

STRUCTURAL_PREFIXES = ("```", "|", "- ", "* ", "+ ", "[源码]", "[测试]", "[设计]")


def heading_topic(heading: str) -> str:
    title = re.sub(r"^#{2,3}\s+", "", heading).strip()
    title = re.sub(r"^\d+(?:\.\d+)*\s+", "", title)
    return title.replace("“", "").replace("”", "")


def classify_first_line(line: str) -> str | None:
    stripped = line.strip()
    if re.match(r"^#{2,4}\s+", stripped):
        return "children"
    if stripped == "```mermaid":
        return "diagram"
    if stripped.startswith("```"):
        return "code"
    if stripped.startswith("|"):
        return "table"
    if re.match(r"^\d+\.\s+", stripped) or stripped.startswith(("- ", "* ", "+ ")):
        return "list"
    if stripped.startswith(("[源码]", "[测试]", "[设计]")):
        return "source"
    if stripped == "---" or not stripped:
        return "other"
    return None


def section_lead(heading: str, kind: str) -> str:
    topic = heading_topic(heading)
    special = {
        "阅读目标": "下面的清单是读完本章后的能力检查，不要求你现在就会。先浏览一遍，阅读正文时再用它确认自己是否抓住主线。",
        "前置知识": "只需具备下面这些基础就可以开始。遇到陌生数学或系统概念时，本章会先给直觉和小例子，再进入源码。",
        "版本与范围": "这一部分先划定结论成立的源码版本和讨论边界。超出范围的实现不能直接套用本章结论，需要重新核对源码。",
        "核心术语": "这张表给出术语在本章中的工作定义。第一次阅读理解含义即可，看到它们在调用链中的位置后再回来看会更清楚。",
        "设计取舍": "下面不只说明代码怎样写，还解释为什么采用这种边界。比较每个方案时，同时考虑正确性、复杂度、显存和性能。",
        "常见误解": "下面集中修正最容易由名称或旧版本经验造成的误解。先判断自己原来的理解，再用后面的源码事实校正。",
        "本章小结": "先用这份清单把本章压缩成一条主线。若某一项还无法用自己的话解释，应回到对应小节和图示，而不是只背结论。",
        "自检问题": "建议先不看答案，用自己的话回答下面的问题。能够解释原因、画出数据流并指出源码位置，才算真正理解。",
        "源码追踪题": "这些题目要求把概念重新落到代码。先从给出的入口搜索定义和调用点，再记录对象、状态与边界，不要只找到同名符号。",
        "参考答案": "参考答案给出的是核对路径，不是唯一措辞。先比较自己的推理在哪一步分叉，再回到对应源码确认。",
        "自检题参考答案": "参考答案用于核对推理过程。重点检查输入、状态变化、输出和边界是否完整，而不是逐句对照文字。",
        "源码与资料索引": "下面按主题整理本章使用的证据入口。需要复核结论时，优先按路径和符号定位，不依赖可能漂移的行号。",
        "源码索引": "下面按主题整理本章使用的证据入口。需要复核结论时，优先按路径和符号定位，不依赖可能漂移的行号。",
        "完成状态": "这里区分正文完成、静态源码核对和真实 GPU 运行验证。没有执行过的实验不会因为正文完整就被标记为已验证。",
        "未解决问题": "这些问题是当前证据边界，不是用猜测补齐的空白。后续应通过指定源码、测试或硬件实验逐项关闭。",
        "建议实验": "下面的实验从单变量和可观察现象开始。运行前先写预期，运行后同时记录环境、结果和限制。",
        "最小验证实验": "先用最小实验验证机制和边界，再扩大模型与负载。功能正确、调用链可见和性能更好是三种不同结论。",
        "阅读与调试指南": "调试时先确定请求停在哪一层，再选择日志、断点或 profile。不要从最底层 kernel 开始漫无目的地搜索。",
    }
    if topic in special:
        return f"> **本节先看：** {special[topic]}"
    if kind == "children":
        body = (
            f"本节要回答：**{topic}**。下面的小节会逐层拆开概念、运行过程和源码落点；"
            "第一次阅读先抓对象之间的关系，第二次再记类名、字段和分支。"
        )
    elif kind == "diagram":
        body = (
            f"这一小节先用图建立“{topic}”的整体路径。先找输入、关键状态和输出，再阅读图后的"
            "源码解释，不必一开始记住全部节点。"
        )
    elif kind == "code":
        body = (
            f"下面的代码或调用链只保留“{topic}”的主干。阅读时依次寻找输入、状态变化和输出，"
            "暂时忽略辅助分支。"
        )
    elif kind == "table":
        body = (
            f"下面先用表格整理“{topic}”。先横向比较每一列解决的问题和适用边界，再把具体名称"
            "映射到源码。"
        )
    elif kind == "list":
        body = (
            f"下面先给出“{topic}”的结论清单。先理解每一项为什么存在，再记参数名或实现细节。"
        )
    elif kind == "source":
        body = (
            f"这一小节主要用源码证据确认“{topic}”。先看符号承担的职责，再追调用关系，不需要"
            "逐行翻译实现。"
        )
    else:
        body = f"先明确“{topic}”要回答的问题和边界，再进入具体实现。"
    return f"> **本节先看：** {body}"


def diagram_guide(body: str, heading: str, index: int) -> str:
    first = next((line.strip() for line in body.splitlines() if line.strip()), "")
    topic = heading_topic(heading) if heading else "本节机制"
    if first.startswith("sequenceDiagram"):
        text = (
            f"这是“{topic}”的时序图。先从左到右确认参与者分别负责什么，再从上到下追踪消息；"
            "第一遍只看正常路径，第二遍再看返回、异步消息和失败分支。"
        )
    elif first.startswith("stateDiagram"):
        text = (
            f"这是“{topic}”的状态图。先找初始状态，再沿箭头观察触发条件和状态变化；重点不是"
            "背状态名，而是弄清谁触发转换、转换后哪些资源需要更新。"
        )
    elif first.startswith("classDiagram"):
        text = (
            f"这是“{topic}”的类型关系图。先分清接口、实现和持有关系，再结合正文确认运行时真正"
            "实例化的是哪个类；类图说明结构，不等同于调用先后。"
        )
    elif first.startswith("xychart"):
        text = (
            f"这是“{topic}”的趋势图。先确认横轴控制变量和纵轴指标，再比较拐点与变化方向；"
            "示意曲线用于解释关系，不能替代在指定硬件和负载上的 benchmark。"
        )
    elif first.startswith("quadrantChart"):
        text = (
            f"这是“{topic}”的二维权衡图。先确认两个坐标轴越大分别意味着什么，再看方案落在哪个"
            "象限；位置表达相对取舍，不表示未经实验验证的精确性能。"
        )
    else:
        direction_match = re.search(r"\bflowchart\s+(LR|RL|TD|TB)\b", first)
        direction = direction_match.group(1) if direction_match else ""
        orientation = {
            "LR": "从左向右",
            "RL": "从右向左",
            "TD": "从上向下",
            "TB": "从上向下",
        }.get(direction, "沿主箭头")
        variants = (
            f"这是“{topic}”的流程图。先{orientation}只追一条主路径，确认输入经过哪些关键阶段到达"
            "输出；第二遍再看虚线、回边和旁路，它们通常表示反馈、复用或可选分支。",
            f"阅读“{topic}”这张流程图时，先把方框看成对象或状态，把箭头看成数据或控制的移动。"
            f"第一遍{orientation}建立顺序，第二遍再核对分支发生的条件。",
            f"这张图用于压缩“{topic}”的整体关系。先{orientation}找到起点、关键转换和终点，再问"
            "每条跨层箭头是否意味着函数调用、消息传递、内存访问或状态更新。",
        )
        text = variants[index % len(variants)]
    return f"> **读图方法：** {text}"


def outside_fence_headings(lines: list[str]) -> list[int]:
    result: list[int] = []
    fence = False
    for index, line in enumerate(lines):
        if line.startswith("```"):
            fence = not fence
            continue
        if not fence and re.match(r"^#{2,3}\s+", line):
            result.append(index)
    return result


def add_chapter_guide(text: str, guide: str) -> str:
    if "## 如何阅读本章" in text:
        return text
    match = re.search(r"^## \d+\.\d+\s+", text, re.MULTILINE)
    if not match:
        raise ValueError("cannot find first numbered section")
    return text[: match.start()] + guide.rstrip() + "\n\n" + text[match.start() :]


def add_section_leads(text: str) -> str:
    lines = text.splitlines()
    insertions: list[tuple[int, list[str]]] = []
    for index in outside_fence_headings(lines):
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            continue
        kind = classify_first_line(lines[cursor])
        if lines[cursor].startswith("> **本节先看：**"):
            following = cursor + 1
            while following < len(lines) and not lines[following].strip():
                following += 1
            if following < len(lines):
                refresh_kind = classify_first_line(lines[following]) or "other"
                lines[cursor] = section_lead(lines[index], refresh_kind)
            continue
        if kind is None or lines[cursor].startswith(">"):
            continue
        insertions.append((index + 1, ["", section_lead(lines[index], kind)]))

    for index, added in reversed(insertions):
        lines[index:index] = added
    return "\n".join(lines) + "\n"


def add_diagram_guides(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    current_heading = ""
    in_mermaid = False
    mermaid_lines: list[str] = []
    diagram_index = 0
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor]
        if not in_mermaid and re.match(r"^#{2,3}\s+", line):
            current_heading = line
        output.append(line)
        if not in_mermaid and line.strip() == "```mermaid":
            in_mermaid = True
            mermaid_lines = []
        elif in_mermaid and line.strip() == "```":
            in_mermaid = False
            diagram_index += 1
            lookahead = cursor + 1
            while lookahead < len(lines) and not lines[lookahead].strip():
                lookahead += 1
            if lookahead >= len(lines) or not lines[lookahead].startswith(
                "> **读图方法：**"
            ):
                output.extend(
                    ["", diagram_guide("\n".join(mermaid_lines), current_heading, diagram_index)]
                )
            else:
                lines[lookahead] = diagram_guide(
                    "\n".join(mermaid_lines), current_heading, diagram_index
                )
        elif in_mermaid:
            mermaid_lines.append(line)
        cursor += 1
    return "\n".join(output) + "\n"


def add_frontmatter_fields(text: str) -> str:
    if "pedagogy_reviewed_at:" in text:
        return text
    marker = "runtime_verified: false\n"
    if marker not in text:
        raise ValueError("cannot locate runtime_verified frontmatter field")
    fields = (
        'audience: "有 LLM 使用经验、代码开发基础和基础数学直觉的工程读者"\n'
        "pedagogy_reviewed_at: 2026-09-07\n"
    )
    return text.replace(marker, marker + fields, 1)


def enrich(path: Path) -> bool:
    chapter = path.parent.name
    original = path.read_text(encoding="utf-8")
    text = add_frontmatter_fields(original)
    text = add_chapter_guide(text, CHAPTER_GUIDES[chapter])
    text = add_section_leads(text)
    text = add_diagram_guides(text)
    if text == original:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed: list[str] = []
    for chapter in CHAPTER_GUIDES:
        path = BOOK / chapter / "README.md"
        original = path.read_text(encoding="utf-8")
        text = add_frontmatter_fields(original)
        text = add_chapter_guide(text, CHAPTER_GUIDES[chapter])
        text = add_section_leads(text)
        text = add_diagram_guides(text)
        if text != original:
            changed.append(chapter)
            if not args.check:
                path.write_text(text, encoding="utf-8")
    if args.check and changed:
        print("Pedagogy enrichment is missing or not idempotent:")
        for chapter in changed:
            print(f"- {chapter}")
        return 1
    action = "would update" if args.check else "updated"
    print(f"Pedagogy enrichment {action}: {len(changed)} chapters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
