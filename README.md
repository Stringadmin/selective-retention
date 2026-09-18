# Selective Retention

> **Temporal and stateful memory for RAG, plus reproducible experiments on selective retention in continual learning.**
>
> **面向 RAG 的时间记忆与状态记忆，以及持续学习中选择性保留的可复现实验。**

[中文](#中文) · [English](#english)

---

## 中文

### 这是什么

这个仓库起初研究的是一个很大的问题：**模型怎样保留重要知识，同时允许自己继续学习？**

我先在参数空间里做了 FIP/GPP：尝试保护已经学到的特征和参数。后来，实验把我带到了信息空间：如果记忆不是模型权重，而是 RAG/Agent 外部存储，那么“什么值得写入”“旧值什么时候失效”“当前状态和历史版本如何共存”仍然是同一个问题的另一种形式。

现在这个仓库主要研究两条相互关联、但不强行混为一谈的路线：

1. **读侧的陈旧证据问题**：RAG 已经召回了新旧事实，为什么 reader 仍然回答旧值？
2. **结构化状态记忆**：对于被明确陈述的事实，如何用写入闸门、当前状态投影和事件归档管理更新？

这不是一个已经被证明优于标准 RAG 的“通用记忆算法”。更准确地说，它是一个带有失败记录的研究代码库：哪些机制在受控条件下有效，哪些机制在自然对话上失效，以及下一步应该把问题放在哪里。

### 先看一个结果

在 LongMemEval-S 的 knowledge-update 题上，我固定了同一组 8 条检索证据，只改变它们呈现给本地 Qwen3-4B reader 的顺序：

| 证据呈现方式 | 有效行准确率 |
|---|---:|
| 相似度排序（现状） | **41.3%** |
| 同一组证据，按时间升序排列 | **65.2%** |
| oracle 证据，按时间升序排列 | **66.7%** |

这里的主要发现不是“时间排序已经解决了 RAG”。它更窄，也更具体：**正确证据进入 top-k 之后，证据的组织方式本身仍然会决定 reader 更可能读到新值还是旧值。** 这个实验只有 72 道题、一个本地 4B reader，也没有使用官方 QA 评测器，所以这些绝对准确率不能外推成通用 RAG 榜单结果。完整方法和限制见 [`docs/LONGMEMEVAL_STALE_LEAK_AND_READER.md`](docs/LONGMEMEVAL_STALE_LEAK_AND_READER.md)。

### 当前研究判断

#### 1. 自然对话和结构化事实不应使用同一种记忆机制

在自然对话中，更新往往不会显式说出它在更新什么属性：

```text
I've tried four different ones so far.
```

这种句子只有放回上下文，才可能知道“four different ones”指的是什么。把一个 slot 抽取器套在句子上，不能解决这个问题。

相反，下面这类内容适合结构化状态记忆：

```text
我的住址是北京。
更新一下，我的住址现在是深圳了。
项目默认使用 pnpm。
```

因此目前的架构判断是：

```text
自然对话 / 情景记忆
    -> 追加式存储 + 语义检索 + 时间顺序组装

明确陈述的事实 / 状态记忆
    -> IGM 写入闸门 + slot + 当前投影 + 版本事件
```

#### 2. IGM 的价值主要是状态语义、压缩和审计，不是合成负载上的准确率魔法

在受控的 500 条 BGE 更新链上：

- 全量版本库配合属性过滤和时间排序：当前值、历史值均为 100%；
- Versioned IGM：当前值、历史值也均为 100%；
- IGM 的差异在于维护一个不包含已取代值的当前投影，同时保留可审计的事件归档。

因此，当前证据支持的说法是：**IGM 是 RAG 前面的结构化事实写入层，不是 RAG 的替代品，也没有在这个合成负载上证明自己比版本化 RAG 更准确。** 它是否值得使用，要看当前状态压缩、prompt token、冲突控制、存储成本和历史查询的取舍。

#### 3. slot 机制有明确边界

在 LongMemEval-S 的 122,416 条自然英文 user turn 上，中文关系模式的英文直译只命中 61 条（0.05%），72 组新旧证据对中没有一组能抽到相同的稳定键。换成本地 4B 模型在写入期逐 turn 抽键后，模糊键复用率为 15.3%。

这不是简单的“正则还不够复杂”。很多自然更新本身没有给出属性名，所以 IGM 当前应该被描述为：

> **面向明确陈述事实的写入层，而不是通用自然对话记忆层。**

#### 4. FIP/GPP 是独立的持续学习研究线

FIP/GPP 验证了一个比较窄但可靠的机制结论：只把梯度置零并不能保证 AdamW 参数不动；权重衰减和历史动量仍然可能造成漂移，真正的冻结需要在优化器更新层投影实际 delta。

但参数级保护在当前合成任务上受到任务间 token 共享的结构性影响，尚不足以支持“同一套重要性保留机制已经从参数空间迁移到记忆空间”的结论。这里保留它，既是研究资产，也是失败边界的记录。

### 30 秒运行一个结构化状态记忆

核心 `igm` 包没有必需依赖：

```bash
pip install -e .
```

```python
from igm import Memory

mem = Memory()
mem.add("我的住址是北京。")
mem.add("更新一下，我的住址现在是深圳了。")

# 当前查询只看到深圳；北京仍在历史事件中
print(mem.query_texts("我现在的住址是什么？"))
print(mem.previous("住址").text)
```

默认语义是**归档式覆盖**：新值成为当前值，旧值关闭有效期但不物理删除。只需要当前状态、不需要历史时，可以显式选择：

```python
Memory(supersede="delete")
```

使用真实嵌入模型时安装可选依赖：

```bash
pip install -e ".[embed]"
```

```python
from igm import Memory, SentenceTransformerEmbedder

mem = Memory(embedder=SentenceTransformerEmbedder("BAAI/bge-small-zh-v1.5"))
```

### 主要复现实验

```bash
# 受控连续更新：版本化语义与当前投影
python -m memory_arch.run_decisive --embed-backend bge \
  --background-facts 24 --full-capacity 28 \
  --random-trials 500 --updates-per-chain 5 --seed 1337 \
  --output reports/memory-decisive-random500-bge.json

# 可解释的写入打分器（弱监督合成标签）
python -m memory_arch.train_scorer

# slot OOD 回归与 Python/JS parity oracle
python -m memory_arch.run_slot_ood \
  --output reports/slot-ood-baseline.json

# LongMemEval-S session 级检索
python -m memory_arch.longmemeval_retrieval \
  --input /path/to/longmemeval_s_cleaned.json \
  --output reports/longmemeval-s-bge-role-aware-test.json \
  --embed-model /path/to/bge-small-zh-v1.5 \
  --partition test --role-aware
```

更多实验命令和参数见 [`docs/`](docs/)。需要本地 BGE 的实验请先下载模型；reader-in-the-loop 实验还需要本地语言模型。所有实验都应结合报告中的数据规模和限制阅读，不建议只复制某一个百分比。

### 研究线状态

| 研究线 | 当前状态 | 我愿意据此声称什么 |
|---|---|---|
| IGM / 结构化事实记忆 | 进行中 | 当前状态、版本事件和写入层边界已经实现并有受控验证 |
| 陈旧证据与时间顺序 | 进行中 | 在一个受限的 LongMemEval-S reader 对照中，时间组装明显改善更新题答案 |
| OutcomeMemory | 暂停 | 结果反馈可用衰减统计解释；事件归档主要提供审计和重算能力 |
| Semantic Context Router | 暂停 | 语义路由可以复用经验，但单 prototype 和 margin 不能作为安全保证 |
| FIP/GPP | 独立研究线 | 优化器层 delta 投影是实现真正冻结的必要机制；当前任务套件未稳定达到原协议门槛 |

### 目录

```text
igm/                    # 零依赖的结构化记忆写入层
  memory.py             # Memory 门面：add / query / history / consolidate
  gate.py               # 重要性闸门与 slot 提取
  store.py              # 事件归档、当前投影、有效期和遗忘

memory_arch/            # Agent 记忆研究实验
  run_decisive.py       # 受控更新与状态/版本消融
  run_stale_leak.py     # 陈旧值在 top-k 内的排序
  run_reader_eval.py    # reader-in-the-loop 顺序对照
  run_write_coverage.py # 自然文本写入覆盖度
  run_llm_keys.py       # 写入期 LLM 抽键
  scorer.py             # 可解释的弱监督写入打分器
  outcome.py            # 结果反馈事件和衰减证据

fip/                    # FIP/GPP 持续学习参数保护
experiments/            # 持续学习实验 runner
dsh-igm-memory/         # DeepSeek Harness 插件
reports/                # 可审计的原始 JSON 结果
docs/                   # 研究报告、长文和架构说明
```

### 边界和诚实声明

- 这个仓库不声称 IGM 普遍优于 RAG，也不声称外部记忆等同于在线训练。
- 合成更新实验的句式和属性池是人为设计的，主要用于验证状态/版本语义，不代表自然用户分布。
- LongMemEval-S reader 实验使用本地 4B 模型，样本量有限，没有使用官方 QA evaluator；目前更适合比较不同臂之间的差异，而不是报告可与榜单直接比较的绝对准确率。
- slot 抽取目前适合“我的 X 是 Y”“请记住 X”“项目约定是 Y”这类明确事实，不适合作为通用对话记忆的前门。
- 结果反馈实验中的可信来源是显式标注的；它不能自动识别伪造的成功信号，也不能替代权限、审核和工具边界。
- 标准 RAG 已经足够好的地方，不应为了使用 IGM 而增加一层系统。

### 详细报告

- [陈旧值泄漏与 reader 评测](docs/LONGMEMEVAL_STALE_LEAK_AND_READER.md)
- [受控更新与公平消融](docs/MEMORY_DECISIVE_LOAD_SWEEP.md)
- [Versioned IGM 架构与验证](docs/VERSIONED_IGM_ARCHITECTURE.md)
- [IGM 研究方案](docs/MEMORY_ARCHITECTURE_RESEARCH_PLAN.md)
- [结果反馈记忆](docs/OUTCOME_FEEDBACK_MEMORY.md)
- [语义情境路由](docs/SEMANTIC_CONTEXT_ROUTING.md)
- [FIP Phase 0 终局决策](docs/FIP_PHASE0_FINAL_DECISION.md)
- [GPP M0 报告](docs/GPP_M0_REPORT.md)
- [配套长文：RAG 找到了新事实，为什么 AI 还是回答旧值？](docs/ARTICLE_DRAFT.md)

### 为什么保留失败结果

我没有把这个仓库整理成一条看起来从未失败过的成功故事。slot 在自然文本上的覆盖失败、语义路由的高置信度错误、OutcomeMemory 与等价统计基线的相同表现，以及 FIP/GPP 没有稳定通过原协议，都是研究结论的一部分。

如果这些结果有价值，它们应该帮助后来者少走一条路，而不是只留下一个漂亮的 README。

---

## English

### What this repository is

This repository started from a broad question: **how can a model retain important knowledge while remaining plastic enough to learn new tasks?**

The first line of work was parameter-level retention in continual learning (FIP/GPP). It then led to a related question in external memory: if knowledge lives in a RAG or agent memory store rather than in model weights, how should we decide what to write, what has become stale, and how current state should coexist with history?

The current work focuses on two related but separate problems:

1. **Stale evidence on the read side**: the retriever returns both an old and a new fact, but the reader still answers with the old one.
2. **Structured state memory**: explicit facts need a write gate, a current-state projection, and an auditable event history.

This is not presented as a universally better memory algorithm than standard RAG. It is a reproducible record of what works under controlled assumptions, what fails on natural dialogue, and where the problem should be moved next.

### The main result

On knowledge-update questions from LongMemEval-S, I kept the same eight retrieved excerpts and changed only their presentation order to a local Qwen3-4B reader:

| Evidence presentation | Answered-row accuracy |
|---|---:|
| Similarity order | **41.3%** |
| The same evidence in chronological order | **65.2%** |
| Oracle evidence in chronological order | **66.7%** |

The claim is deliberately narrow: **after the right evidence has been retrieved, evidence organization can still determine whether the reader follows the current or the stale value.** This was one reader, 72 questions, and a non-official answer evaluator. The numbers should be read as an intervention result, not as a general RAG benchmark.

See [`docs/LONGMEMEVAL_STALE_LEAK_AND_READER.md`](docs/LONGMEMEVAL_STALE_LEAK_AND_READER.md) for the protocol, ablations, and limitations.

### The current architectural view

Natural conversation and explicit state should not be forced through the same memory mechanism.

Natural conversation often leaves the attribute implicit:

```text
I've tried four different ones so far.
```

A slot extractor cannot know what “ones” refers to without recovering context across turns. In contrast, explicit statements such as the following are suitable for structured state memory:

```text
My address is Beijing.
My address is now Shenzhen.
The project uses pnpm by default.
```

The working split is therefore:

```text
Natural / episodic memory
    -> append-only evidence + semantic retrieval + temporal assembly

Explicit state facts
    -> IGM write gate + slot + current projection + versioned events
```

### What IGM does, and does not do

IGM is a write layer in front of RAG. It provides:

- an importance gate for candidate memories;
- optional slot extraction for explicit facts;
- an event archive with validity intervals;
- a current-state projection that excludes superseded values;
- history lookup and a small forgetting lifecycle.

On the controlled 500-chain BGE workload, both a version-aware baseline and Versioned IGM reached 100% on current and historical queries. The evidence supports a semantic and systems claim—not an accuracy win over versioned RAG:

> IGM is useful when explicit state, conflict isolation, compression, and auditability matter. It is not a replacement for RAG.

Natural-language coverage is a known boundary. A literal English translation of the shipped slot patterns fired on only 61 of 122,416 user turns (0.05%) and produced no shared key across 72 update pairs. A local 4B model raised fuzzy key reuse to 15.3%, but could not recover attributes that were never named in the source turn.

### FIP/GPP as a separate research line

FIP/GPP established a narrower mechanism result: setting gradients to zero is not sufficient to freeze AdamW parameters. Decoupled weight decay and stale optimizer momentum can still move them; freezing the actual optimizer delta is required.

The current continual-learning task suite also contains substantial cross-task token sharing, and the original protocol threshold was not met stably. I therefore keep FIP/GPP as a separate research line rather than claiming that the same retention mechanism has already transferred from parameter space to memory space.

### Quick start

The core `igm` package has no required dependencies:

```bash
pip install -e .
```

```python
from igm import Memory

# The shipped default slot patterns are Chinese explicit-fact patterns.
# Replace the gate with a language-specific implementation for English data.
mem = Memory()
mem.add("我的住址是北京。")
mem.add("我的住址现在是深圳。")

# The current view contains Shenzhen; Beijing remains queryable as history.
print(mem.query_texts("我现在的住址是什么？"))
print(mem.previous("住址").text)
```

The default policy archives superseded events instead of deleting them. If only the current state is needed:

```python
Memory(supersede="delete")
```

For real embeddings:

```bash
pip install -e ".[embed]"
```

```python
from igm import Memory, SentenceTransformerEmbedder

mem = Memory(embedder=SentenceTransformerEmbedder("BAAI/bge-small-zh-v1.5"))
```

### Research status

| Line | Status | Supported claim |
|---|---|---|
| IGM / structured fact memory | Active | Current state, version events, and write-layer boundaries are implemented and tested |
| Stale evidence and temporal order | Active | Chronological assembly improved update-question answers in a controlled reader comparison |
| OutcomeMemory | Paused | Decayed evidence adapts; the event archive mainly adds auditability and replay |
| Semantic context routing | Paused | Semantic routing can reuse experience, but a single prototype and margin are not safety guarantees |
| FIP/GPP | Separate line | Optimizer-level delta projection is necessary for true freezing; the current task suite did not stably meet the original gate |

### Limitations

- This project does not claim that IGM generally outperforms RAG.
- The synthetic update loads use hand-designed templates and value pools. They validate state/version semantics, not natural user distributions.
- The LongMemEval-S reader study uses one local 4B model, a limited sample, and a non-official evaluator. It is best interpreted through arm-to-arm differences.
- The shipped slot extractor is for explicit fact statements, not general conversational memory.
- Outcome provenance is explicitly provided by the system in the experiments; it is not an automatic defense against forged feedback or prompt injection.
- If standard RAG already works well for a task, adding this layer is unnecessary.

### Detailed reports

- [Stale evidence and reader evaluation](docs/LONGMEMEVAL_STALE_LEAK_AND_READER.md)
- [Controlled update load sweep](docs/MEMORY_DECISIVE_LOAD_SWEEP.md)
- [Versioned IGM architecture](docs/VERSIONED_IGM_ARCHITECTURE.md)
- [Memory architecture research plan](docs/MEMORY_ARCHITECTURE_RESEARCH_PLAN.md)
- [Outcome feedback memory](docs/OUTCOME_FEEDBACK_MEMORY.md)
- [Semantic context routing](docs/SEMANTIC_CONTEXT_ROUTING.md)
- [FIP Phase 0 final decision](docs/FIP_PHASE0_FINAL_DECISION.md)
- [GPP M0 report](docs/GPP_M0_REPORT.md)

### Why the negative results remain here

I do not want this repository to read like a success story in which every mechanism worked. The failure of surface slot extraction on natural text, high-confidence routing errors, the equivalence between OutcomeMemory and a decayed-statistics baseline, and the unstable FIP/GPP gate are all part of the research record.

If the repository is useful, it should help someone avoid an unproductive experiment—not merely provide a polished claim.

---

## License

[MIT](LICENSE)
