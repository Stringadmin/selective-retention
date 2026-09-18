# RAG 找到了新事实，为什么 AI 还是回答旧值？

## 我在参数保护、记忆写入和证据排序上连续撞墙后的研究记录

> **English title:** *Why does RAG answer the old fact when it has already retrieved the new one?*
>
> This is a research log about continual learning, structured agent memory, and stale evidence. It does not claim a universal replacement for RAG.

### English abstract

I started with a simple goal: make an AI system retain important knowledge while continuing to learn. The project moved through three layers: protecting parameters in continual learning, deciding what to write into agent memory, and ordering conflicting evidence at read time.

The most reliable low-level result is easy to state: zeroing gradients does not freeze AdamW parameters. Decoupled weight decay and optimizer momentum can still move them, so the actual optimizer delta must be projected at the update layer.

The memory results were less flattering and more useful. On controlled update chains, IGM (Importance-Gated Memory) correctly maintains a current-state view and an event archive, but it does not beat a strong version-aware RAG baseline on accuracy. On LongMemEval-S knowledge-update questions, the stronger signal came from the reader side: keeping the same retrieved evidence but presenting it chronologically improved a local Qwen3-4B reader from 41.3% to 65.2% effective accuracy. Meanwhile, surface-form slot extraction almost completely failed on natural dialogue.

The current conclusion is therefore narrower than the original idea: explicit facts can use structured state/version memory; ordinary conversation needs temporal evidence assembly; and these two kinds of memory should not be forced through one write rule.

---

## 先把结论放在前面

我最初想研究的是“怎么让 AI 不遗忘”。现在我会把问题拆成三个层次：

1. **参数层**：训练新任务时，怎样真的保护旧参数？
2. **状态层**：一个明确的事实更新后，当前值和历史版本怎样共存？
3. **证据层**：新旧事实都被检索出来时，怎样把证据交给 reader，避免它跟着旧值走？

这三个问题有相似的直觉，但不是同一个问题。我一开始把它们都叫“选择性保留”，后来才发现，真正重要的不是给所有问题套上同一个机制，而是先判断**约束到底应该在哪一层生效**。

现在项目里最值得继续研究的结果，是第三层：

| 证据呈现方式 | LongMemEval-S knowledge-update，有效行准确率 |
|---|---:|
| 相似度排序 | **41.3%** |
| 同一组证据按时间升序排列 | **65.2%** |
| oracle 证据按时间升序排列 | **66.7%** |

这里的证据集合相同，只改变呈现顺序。实验只有 72 道题、一个本地 Qwen3-4B reader，也没有使用官方 QA 评测器，所以这不是通用 RAG 榜单结果。它只说明一件比较具体的事：**召回正确证据以后，证据如何组织，仍然可能决定模型回答新值还是旧值。**

下面是这条结论是怎么撞出来的。

---

## 一、梯度已经是零了，参数为什么还在动？

第一个方向是持续学习里的灾难性遗忘。

最自然的想法是：新任务训练时，找出承载旧知识的重要特征，把它们保护起来。我的第一版 FIP（Feature Isolation and Plasticity）把 FFN 中间特征分成 stable、shared、plastic 和 free 几种状态，再通过梯度 hook 控制它们的更新。

思路很简单：

```text
重要特征 -> stable
stable 特征 -> 梯度清零
梯度为零 -> 参数不再更新
```

我还加了 Top-K 稀疏激活、任务结束后的重要性固化、状态机和回滚。单元测试通过以后，我做了一个最基本的 sanity check：故意冻结一批参数，只把梯度清零，训练几十步，然后检查参数漂移。

结果如下：

| 冻结方式 | 冻结参数最大漂移 |
|---|---:|
| 梯度 hook，梯度置零 | `9.94e-05` |
| 优化器 delta 投影 | `0.00` |

不是测试错了。问题在 AdamW 的更新并不等于“当前梯度乘学习率”。

### 两个我漏掉的更新通道

第一是解耦权重衰减：

```text
parameter <- parameter * (1 - lr * weight_decay)
```

这一步不依赖当前梯度。梯度被清零以后，权重仍然会被往零拉。

第二是优化器状态。Adam 保存了过去梯度的滑动平均，即使当前梯度为零，历史动量仍然可能继续推动参数。

所以我之前做的事情，本质上是把一个输入信号关掉了，却没有关掉真正执行更新的地方。我以为门锁了，实际上只是把门铃拔了。

### 修复：在更新层投影真实 delta

后来我把保护逻辑放进 `FeatureMaskedAdamW`：先让 AdamW 计算它原本想做的更新，再对实际参数增量做投影：

```text
parameter_after = parameter_before + mask * optimizer_delta
```

stable 参数的 mask 为 0，shared 参数可以使用较小的缩放系数，其他参数保留正常更新。同时清理被冻结切片对应的历史动量。

这样以后，约束作用在了真正改变参数的地方，而不是作用在优化器之前的一个中间信号上。

### 我从这里得到的教训

> **“我把控制信号关掉了”和“系统真的无法违反这个约束”，不是一回事。**

任何需要强制执行的机制都应该问一句：最终状态在哪里被改变？如果答案是优化器，就不能只在梯度层做文章。

但这只是证明了“如何真正冻结参数”。它没有证明冻结参数就一定能解决灾难性遗忘。

---

## 二、机制没有错，但我保护的地方不是瓶颈

修好冻结以后，我按预先冻结的协议跑了完整矩阵：普通 dense 微调、稀疏全量微调、EWC、回放、FIP 和后续的 GPP 变体。

早期实验一度给出比较乐观的局部结果，但在最终的 batch 32 口径下，FIP 的核心对照是：

| 方案 | 平均遗忘 |
|---|---:|
| 稀疏全量微调基线 E | `0.4555` |
| FIP，重要性分配 | `0.5038` |
| FIP，随机分配 | `0.6782` |
| FIP + 累积式非 FFN 保护 | `0.3952` |

这里有一个重要但容易被忽略的结果：重要性分配比随机分配好约 25.7%。所以“重要性分配完全没有用”不是结论。

但 FIP 仍然没有达到原协议要求的“相对稀疏基线降低遗忘 50%”。继续做 drift 分析后发现，遗忘的主要来源并不只在 FFN：embedding、attention、norm，以及任务间共享的 token 都在参与问题。

GPP 把保护范围扩大到非 FFN 参数以后，机制效果有所改善：三种子平均遗忘约为 `0.296`，优于 dense 和稀疏基线；但原协议门槛是 `0.2278`，仍然没有稳定通过。尤其是 modular 与 shared 任务共享 NUMBER token，冻结一边的参数并不能阻止另一边通过共享结构造成干扰。

### 任务设计不是一个小细节

当前合成任务里有明显的跨任务 token 共享：

- facts 和 conflict 共享 entity token；
- modular 和 shared 共享 number token；
- string 和 unrelated 共享 source token。

这让“保护旧任务”与“允许新任务学习”之间出现了结构性冲突。继续增加更复杂的冻结策略，并不能自动消除这个冲突。

### 我从这里得到的教训

> **机制没有绝对好坏，先要确认它是不是打在真正的瓶颈上。**

我当时很容易爱上自己的机制：重要性、特征槽、冻结、回滚，这些东西都很漂亮。但如果主要遗忘路径在非 FFN 参数，或者数据本身让任务共享了输入结构，那么 FFN 保护再精巧也只能局部有效。

这也是为什么现在我不把 FIP/GPP 和 IGM 包装成“同一个方法已经跨空间迁移成功”。目前更准确的说法是：它们共享一种研究直觉，但迁移本身还没有被证明。

---

## 三、把选择性保留搬到 Agent 记忆：先输给了简单 RAG

参数保护做不顺以后，我把问题换了一个空间。

Agent 的外部记忆也有类似矛盾：

```text
用户以前住北京。
用户现在住深圳。
```

如果把这两句话直接追加进向量库，它们的语义很接近。一个只会做向量 top-k 的系统可以把两条都找出来，却不会自动知道哪条是当前值、哪条是历史值。

于是我做了 IGM（Importance-Gated Memory）：

```text
候选文本
  -> 重要性闸门
  -> slot 提取
  -> 当前状态投影 / 事件归档
  -> 向量检索或 slot 路由
```

![RAG 与 IGM 写入层架构对比](fig1_rag_vs_igm.png)

IGM 从一开始就不是 RAG 的替代品。它只负责 RAG 的写入侧：决定候选内容是否进入记忆，以及明确的同属性更新如何处理。

### IGM 的三个组件

#### 1. 写入闸门

候选记忆先经过一个重要性评分器。当前实验里的 scorer 是可解释的 logistic regression，但训练标签来自受控弱监督：

- “我的 X 是 Y”一类的持久事实作为正例；
- 闲聊、问句和瞬时事件作为负例。

这能验证门控逻辑和写入成本，但不能证明模型已经学会了真实的“未来价值”。held-out accuracy 主要说明它识别合成分布里的事实模板，不应该被解读为通用记忆重要性预测器。

#### 2. slot 和当前状态

对于明确陈述的事实，从句子里提取属性名作为 slot：

```text
我的住址是北京。
我的住址现在是深圳。
```

第二句会关闭第一个事件的当前有效期，让深圳成为当前投影里的唯一值。

#### 3. 事件归档

最开始的 IGM 是破坏性覆盖：旧值直接删掉。它能回答当前问题，却无法回答“之前是什么”。后来我把语义改成默认的归档式覆盖：旧事件保留在 archive 里，当前投影只暴露仍然有效的事件。

事件大致包含：

```text
text
slot
event_id
valid_to
supersedes
created_at
```

当前投影由 `valid_to is None` 的事件派生，`supersedes` 保留版本之间的关系，而不是再维护一份容易和归档分叉的独立指针表。这就是现在 `igm/store.py` 和 `dsh-igm-memory` 插件使用的 Versioned IGM 语义。

---

## 四、IGM 没有赢过版本化 RAG，但它把产品语义说清楚了

我先做了一个受控更新实验：同一个属性连续更新 5 次，加入 24 条需要长期保留的背景事实，再问当前值和历史值。

每个 seed 跑 500 条随机化链，使用同一批 BGE、同一套输入和同一轮门控决定。结果如下：

| 方法 | 当前值 | 历史值 | 持久化事件/记录 | 当前投影 |
|---|---:|---:|---:|---:|
| 原始全量，语义 top-1 | 22.8% / 23.2% | - | 52 | 52 |
| 原始全量，属性 + 时间 | 100% / 100% | 100% / 100% | 52 | 52 |
| 门控但保留版本，属性 + 时间 | 100% / 100% | 100% / 100% | 32 | - |
| 破坏性 IGM | 100% / 100% | 0% / 0% | 28 | 28 |
| **Versioned IGM** | **100% / 100%** | **100% / 100%** | **32** | **28 个事件引用** |

这个结果把原本的宣传口径纠正了：

> **IGM 没有在这个负载上证明自己比版本化 RAG 更准确。**

一个带有属性字段、时间字段和二级索引的版本库，也可以在读取当前值时只取最新记录。IGM 真正提供的是一套统一的接口和语义：

- 当前状态是什么；
- 历史版本在哪里；
- 哪个事件取代了哪个事件；
- 当前读路径不应该看到哪些旧值；
- 事件什么时候可以被 consolidate 或 prune。

如果产品只需要当前状态，破坏性覆盖可以减少当前投影的条目数；如果产品需要审计和历史查询，归档式覆盖更安全，但事件库会随更新次数增长。

这是一种产品语义和工程取舍，不是一个自动产生准确率优势的魔法。

![受控连续更新实验的机制示意](fig2_decisive_experiment.png)

图中展示的是原始手工更新链的机制示意，不是 500 条随机化负载的统计图。

---

## 五、真正意外的结果：问题可能在读侧，而不是写侧

在受控负载里，slot 已经把“哪个属性被更新”告诉了系统。为了看看这个假设在自然数据上是否成立，我转向 LongMemEval-S 的 knowledge-update 题。

英文 BGE 的结果是：

- 正确的新证据几乎总能进入 top-10，`recall_all@10` 为 98.31%；
- 但在 72 道有新旧两条证据的题里，旧值排在新值前面的比例约为 61.1%；
- 也就是说，召回指标看起来很好，但 reader 仍然可能先读到陈旧值。

我随后让本地 Qwen3-4B reader 读取同一组证据，只改变证据的呈现顺序：

| 臂 | 说明 | 有效行准确率 |
|---|---|---:|
| `stale_only` | 只给旧值 | 4.4% |
| `reverse_k8` | 同一组证据，最新值放前面 | 13.8% |
| `ranked_k8` | 相似度排序，当前读侧策略 | 41.3% |
| `current_only` | 只给当前值，模拟完美写入层 | 58.8% |
| `chrono_k8` | 同一组证据按时间升序 | 65.2% |
| `oracle` | 标注证据按时间升序 | 66.7% |

两个结果让我重新看待 IGM：

1. **只保留当前值并不是免费的最优解。** 在这个 reader 和这个负载上，`current_only` 反而低于保留上下文、再按时间组装的 `chrono_k8`。
2. **读侧的时间组织可能比写侧的删除更便宜。** 不需要重新抽键，也不需要修改向量库，只利用已有时间戳改变 prompt 组装方式，就获得了约 24 个百分点的差异。

当然，这仍然不是“时间排序对所有模型都有效”的证明。reader 只有一个，数据量也不大。后续还需要多个模型、更多题型以及对语篇连贯性和位置效应的控制实验。

但这是目前项目最值得继续投入的研究线，因为它提出了一个标准 RAG 评测经常看不到的问题：

> **检索到了正确证据，不等于 reader 会以正确的顺序理解证据。**

---

## 六、slot 在自然对话上为什么救不回来

如果写入层要处理自然对话，最直接的想法是：换一个更强的抽取器，让 LLM 帮忙给每句话命名 slot。

我做了两个检查。

### 表层规则

把中文规则直译成英文，放到 LongMemEval-S 的 122,416 条 user turn 上：

- 中文规则在英文自然文本上几乎不触发；
- 英文直译规则只命中 61 条，命中率 0.0498%；
- 72 组新旧证据对中，旧值和新值没有一组抽到同一个稳定键。

### LLM 抽键

让本地 Qwen3-4B 在写入期逐条读取 user turn，不看问题，也不看另一处提及，然后抽取一个候选属性名：

- 精确或近似可复用键：11/72，约 15.3%；
- 43% 的证据对至少有一侧完全抽不出键；
- 放宽相似阈值以后，复用率会上升，但不同属性被误合并的风险也开始上升。

根因并不是“LLM 还不够大”。很多更新本身就没有在当前句子里命名属性：

```text
I've tried four different ones so far.
```

如果不看上下文，它不可能知道“ones”指的是餐厅、方案、设备还是别的东西。

所以当前结论是：

> **slot 机制适合作为明确事实库的写入层，不适合作为所有自然对话的通用记忆前门。**

这反而帮助我把架构边界画清楚了。

---

## 七、现在比较合理的架构：两种记忆，两条路径

我不再试图用一套规则处理所有记忆。

### A. 结构化状态记忆

适用于：

- 用户明确要求“记住”的事实；
- 项目配置和约定；
- 当前地址、默认语言、工具链；
- 明确的偏好、决策和状态更新。

它可以使用：

```text
写入闸门
  -> slot
  -> 当前状态投影
  -> 版本事件归档
  -> current / history 两类读出口
```

### B. 非结构化情景记忆

适用于：

- 普通对话；
- 尚未命名的偏好；
- 需要跨句理解的事实；
- “之前那件事”“另一个方案”这类指代；
- 很难在写入当下确定属性名的内容。

它更适合：

```text
追加式证据
  -> 语义检索
  -> 时间和会话顺序组装
  -> reader 判断当前语义
```

这两条路径不一定需要两个完全不同的数据库，但需要不同的语义契约。明确状态可以覆盖；自然情景不应该在没有足够置信度时自动覆盖旧证据。

这可能是比“做一个万能 importance scorer”更现实的研究方向：**先判断记忆是什么，再决定它如何被保存和读取。**

---

## 八、三次撞墙后，我现在真正愿意声称什么

### 1. 约束必须在执行层生效

在参数空间里，梯度 hook 不足以冻结 AdamW；需要投影真实 optimizer delta。

在记忆空间里，写入层可以定义当前状态，但读侧同样可能通过错误的证据顺序重新引入陈旧值。

所以“执行层”这个经验仍然成立，但不能被简单地理解成“所有问题都应该在写入层解决”。真正的问题是：**哪个层最终决定系统行为？**

### 2. 重要性保留不是通用胜利条件

FIP 的智能分配比随机分配好，IGM 在显式事实的受控负载上可以正确维护当前状态，但这些结果都不等于“选择性保留已经解决了持续学习或 Agent 记忆”。

重要性信号是否有用，取决于：

- 信号是否真的能预测未来复用；
- 数据是否允许稳定抽取对象和属性；
- 瓶颈在写入、存储、召回还是 reader；
- 基线是否已经用时间、属性和版本字段解决了问题。

### 3. “参数空间和信息空间是同一套机制”目前仍然是假说

FIP/GPP 和 IGM 的确有相似结构：

```text
重要性信号
  -> 选择性保留
  -> 在执行层强制应用
```

但目前的实验证据只支持“方法论上相似”，不支持“机制已经被证明可迁移”。这句话我现在会明确写在项目边界里，而不会把它当作结论。

---

## 九、哪些方向我暂时不再继续推进

### OutcomeMemory

结果反馈记忆可以在有时间衰减和概念漂移的受控环境里恢复当前更优动作，也可以通过来源权重降低**已明确标记**的低可信反馈影响。

但它与等价的衰减统计基线完全一致。事件归档的价值是审计、回放和重算，不是它自己产生了额外的预测能力。要继续做，下一步必须接入真实工具任务，而不是继续扩大合成 bandit 规模。

### Semantic Context Router

语义路由能把新措辞映射到已有情境，但单 prototype 和 top-1/top-2 margin 都不能作为可靠安全闸门。留出集上仍然存在高置信度错路由。没有真实任务和澄清/回退流程之前，这条线不值得继续调阈值。

### 把 slot 规则扩展成通用自然语言解析器

自然更新经常缺少属性名。继续堆更多正则，或者换一个模型抽键，都不能解决“源句没有提供足够信息”这个问题。更合理的路径是允许跨句推断、引入问题和会话上下文，或者干脆把这类内容保留为情景证据，不强行压缩成 slot。

---

## 十、代码和复现

安装核心库：

```bash
pip install -e .
```

运行受控更新实验：

```bash
python -m memory_arch.run_decisive --embed-backend bge \
  --background-facts 24 --full-capacity 28 \
  --random-trials 500 --updates-per-chain 5 --seed 1337 \
  --output reports/memory-decisive-random500-bge.json
```

训练弱监督写入打分器：

```bash
python -m memory_arch.train_scorer
```

运行 slot OOD 回归：

```bash
python -m memory_arch.run_slot_ood \
  --output reports/slot-ood-baseline.json
```

LongMemEval-S 的完整检索、陈旧值泄漏、reader 评测和 LLM 抽键协议见：

- [`docs/LONGMEMEVAL_STALE_LEAK_AND_READER.md`](LONGMEMEVAL_STALE_LEAK_AND_READER.md)
- [`docs/MEMORY_DECISIVE_LOAD_SWEEP.md`](MEMORY_DECISIVE_LOAD_SWEEP.md)
- [`docs/VERSIONED_IGM_ARCHITECTURE.md`](VERSIONED_IGM_ARCHITECTURE.md)

IGM 也可以作为 Python 库使用；DeepSeek Harness 插件位于 [`dsh-igm-memory/`](../dsh-igm-memory/)。

---

## 最后

我没有做出一个能在所有持续学习和 Agent 记忆任务上刷榜的系统。

但我现在比项目开始时更清楚三件事：

- 梯度为零不代表参数真的被冻结；
- 记忆写入正确不代表 reader 会正确处理新旧证据；
- 自然对话、结构化状态和历史事件，不应该被压缩成同一种记忆语义。

如果这项工作有一点价值，我希望它不是让人相信 IGM 已经解决了 Agent 记忆，而是让后来者在开始实验前先问几个更具体的问题：

```text
我到底在保护参数、状态，还是证据？
基线是否已经有时间和版本语义？
问题真正的瓶颈在写入、召回，还是 reader？
这个句子里真的存在可以被抽取的属性吗？
```

这些问题比再加一个“重要性分数”更可能决定实验有没有意义。

所有代码、数据生成器、评估脚本和原始 JSON 结果都在 GitHub：

[Stringadmin/selective-retention](https://github.com/Stringadmin/selective-retention)

项目使用 MIT License。
