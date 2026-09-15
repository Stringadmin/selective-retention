# igm — RAG 的写入层

一个轻量、零依赖的 Python 库，给 RAG 记忆系统补上它缺失的**写入层**。

标准 RAG 管"读"（检索），但不回答两个写的问题：**什么值得存？存的东西过时了怎么办？** 在向量库里，"用户住北京"和"用户现在住深圳"是两条平级向量，没有新旧关系——这就是知识更新/矛盾解决失败的根源。

igm 在写入侧做两件事：

- **重要性闸门**：只把值得记的写进去（闲聊、问句被过滤）。
- **slot 覆盖**：同一属性的新值成为唯一当前值。旧值被关闭有效期后留在事件归档里——读侧看不到，历史仍可查。

## 安装

```bash
pip install -e .            # 核心零依赖
pip install -e ".[embed]"   # 需要真实嵌入（BGE 等）时
```

## 30 秒上手

```python
from igm import Memory

mem = Memory()                              # 零依赖默认配置
mem.add("我的住址是北京。")
mem.add("更新一下，我的住址现在是深圳了。")    # 关闭北京那条的有效期
mem.query_texts("我现在的住址是什么？")        # -> ['...现在是深圳了。']，不含北京
mem.previous("住址").text                   # -> 北京那条，历史仍可答
len(mem), mem.store.event_count             # -> 1, 2   （当前投影 / 含归档）
```

只关心当前状态、不想让事件库增长时，选破坏性覆盖：

```python
Memory(supersede="delete")   # 旧值物理删除，历史不可查
```

闲聊和问句会被自动过滤：

```python
mem.add("今天天气不错。")        # 不写入
mem.add("我的爱好是什么？")      # 问句不是记忆，不写入
mem.add("我的爱好是摄影。")      # 写入
```

## 用真实嵌入模型

默认的 hash 嵌入只够演示。生产环境接真实嵌入（任何 `(str)->list[float]` 都行）：

```python
from igm import Memory, SentenceTransformerEmbedder

mem = Memory(embedder=SentenceTransformerEmbedder("BAAI/bge-small-zh-v1.5"))
```

或者接你自己的嵌入服务（OpenAI / Cohere / 自部署）：

```python
from igm import Memory, CallableEmbedder
mem = Memory(embedder=CallableEmbedder(my_embed_fn))
```

## 接进你的 RAG

igm 不动你的向量库和检索，只在写入前插一层：

```python
# 之前:  for turn in conversation: vectordb.add(turn)
# 之后:
mem = Memory(embedder=your_embedder)
for turn in conversation:
    mem.add(turn)            # 闸门筛选 + slot 覆盖
# 检索时
context = mem.query_texts(question, top_k=5)
```

记忆量大时，定期巩固（遗忘低价值记忆）：

```python
mem.consolidate()   # 会话/天边界调用
```

## 核心机制的效果

同一属性连续更新 5 次（新旧值语义近似），问"当前值"：

| 方法 | 准确率 | 记忆用量 |
|---|---:|---:|
| 标准 RAG（全量写入） | 0.00 | 28 条 |
| 标准 RAG + 时间排序 | 0.60 | 28 条 |
| **RAG + igm 写入层** | **1.00** | **4 条** |

"记忆用量"数的是当前投影。默认策略下事件归档另外保留被取代的版本（该负载里那条属性链留 5 个事件），代价见根目录 README 的边界一节；`supersede="delete"` 时归档为空。

完整证据与复现见仓库根目录 README 和 `docs/ARTICLE_DRAFT.md`。

## 什么时候别用

- 你的场景标准 RAG（全量 + 强嵌入）就够好 —— 那不需要这层。
- 只是检索不准 —— 先换个好嵌入，别动写入层。
- igm 是 RAG 的**增强件**，不是替代品。

## License

MIT
