# Versioned IGM：状态视图与版本事件库

> 状态：已下沉为 `igm` 与 `dsh-igm-memory` 插件的默认语义（2026-09-11）。本页保留原始验证与其边界，迁移差异见最后一节。

## 动机

破坏性 slot 覆盖把“当前状态”误当作“全部记忆”：当前问题回答正确，历史问题却无从回答。版本化存储改为两个逻辑层，但不复制事实文本或嵌入：

```text
候选事实
  -> IGM 门控与 slot 抽取
  -> append-only 事件归档（MemoryItem，每条带 valid_to / supersedes）
  -> 当前投影（valid_to 为空的事件）

当前问题  -> current(slot)   -> 该 slot 的唯一当前事件
历史问题  -> history(slot)   -> 该 slot 的版本时间线
```

同一 slot 写入新值时，旧事件的 `valid_to` 被关闭，新事件记录
`supersedes=<old event id>`；旧事件继续留在归档。破坏性覆盖降级为显式选项
`MemoryStore(supersede="delete")`，仍是原来的语义。

## 与原始原型的实现差异

`memory_arch/versioned.py`（`VersionedMemoryStore` / `VersionedEvent`）已删除，其逻辑成为
`igm/store.py` 的默认实现。两点有意偏离原型：

- **当前投影按 `valid_to is None` 派生**，不再有 `current_by_slot` 指针表。原型那张表需要与
  `prune` 同步维护，一旦遗忘路径漏更新就会出现"指针指向已删除事件"的不一致；派生扫描是 O(n)，
  而检索本来就是 O(n)，在这个参考实现的规模下不换这点复杂度。
- **事件即条目**：`MemoryItem` 直接携带 `valid_to` / `supersedes` / `event_id`，不再包一层
  `VersionedEvent`。因此 `memory_arch` 也不再保留自己那份 store 拷贝，改从 `igm` 导入，
  库与实验框架共用同一实现。

## 受控验证

与现有 500 条随机化 BGE 负载使用相同的输入、BGE、门控分数与 slot 抽取器。每条链有 5 次同属性更新、24 条背景事实；seed `1337` 和 `42` 各运行 500 条。答案仍由规则读取存储，以测量记忆层而不混入 LLM 推理随机性。

| 方法 | 当前值，1337 | 当前值，42 | 历史值，1337 | 历史值，42 |
|---|---:|---:|---:|---:|
| 门控但保留版本，属性 + 时间 | 500/500 | 500/500 | 500/500 | 500/500 |
| 破坏性 IGM（slot 覆盖） | 500/500 | 500/500 | 0/500 | 0/500 |
| **Versioned IGM** | **500/500** | **500/500** | **500/500** | **500/500** |

每条链的 Versioned IGM 写入 32 个门控事实事件；当前投影含 28 个活跃 slot 引用，不复制事件；按明确 slot 路由的当前问题只解引用 1 个事件，历史问题查看该属性的 5 个版本。24 条固定背景事实召回为 `24/24`。

原始输出：

- `reports/memory-decisive-random500-bge.json`（seed 1337）
- `reports/memory-decisive-random500-bge-seed42.json`（seed 42）
- `reports/memory-versioned-igm-random500-bge.json`（seed 1337，迁移前 versioned 臂）
- `reports/memory-versioned-igm-random500-bge-seed42.json`（seed 42，迁移前 versioned 臂）
- `reports/memory-decisive-random500-bge-postversioned.json`（seed 1337，迁移后复跑）
- `reports/memory-decisive-random500-bge-postversioned-seed42.json`（seed 42，迁移后复跑）

迁移后复跑（2026-09-14，WSL `fip-venv`，BGE-small-zh-v1.5，RTX 5070）确认数字不变：两个 seed 的
全部共享指标与迁移前**逐位一致**——`full_top1` 0.228 / 0.232、`full_recency` 0.668 / 0.714、
破坏性臂 `igm` 1.00 / `igm_history` 0.00、归档臂 `versioned_igm` 1.00 / `versioned_igm_history`
1.00、背景事实召回 24/24；每链 `32 事件 / 28 当前` 与原 versioned 报告相同。同配置的 hash 负载
（`reports/memory-decisive-random500-hash-postversioned.json`）也复现为 `1.00 / 0.00 / 1.00 / 1.00`。
唯一差异是运行形态：迁移前 versioned 臂单独成报告，现在两臂同置于 `run_decisive` 的一个报告里，
破坏性语义通过 `supersede="delete"` 显式表达。

## 结论与边界

这个实验验证的是实现不再牺牲历史查询，**不是** Versioned IGM 比版本化 RAG 更准。门控但保留版本的强基线也同样答对当前和历史；一个支持二级索引的版本库也可将当前查询路由到单条事件。

Versioned IGM 的工程价值只在于把以下策略统一成明确接口：门控、规范化事实事件、当前状态投影、版本有效期与查询意图路由。它是否值得引入，仍取决于实际 Agent 的 prompt token、索引/存储成本、延迟、抽取错误和历史查询比例。

下一阶段必须在真实或公开基准上比较：版本化 RAG、数据库 upsert、TTL/LRU、Versioned IGM；同时报告当前/历史准确率、冲突率、prompt token、持久化字节和 p95 读写延迟。不能再用模板化合成链推导通用 RAG 结论。

## 插件侧同步（`dsh-igm-memory`）

`IgmStore` 用同样的三个字段实现这一语义：`validTo`（`null` 即当前有效）、`eventId`、`supersedes`。
`current()` 是当前投影，`query`、`recall_fact`、跨项目经验匹配和会话启动注入都只走它，因此被取代的
值不会再进入模型上下文；`size` 数当前投影，`eventCount` 数归档。旧版 JSON 存储缺这三个字段，
`load()` 会把它们补齐并按顺序分配 `eventId`，历史记录一律视为当前有效——第一次更新即完成升级，
由测试 `a v1 store file upgrades to versioned supersede on first update` 钉住。

迁移过程中发现并修掉一个真实缺陷：`add()` 的 text-dedup 分支原先先按 **slot** 查找已有记录，
命中即就地改写文本，于是同属性的更新永远走不到下面那行 `items.filter(slot !== ...)` 覆盖逻辑——
那行是死代码，旧值是被 dedup 原地销毁的。现在 dedup 只认完全相同的文本（原样重述不算新版本），
同 slot 的不同措辞按版本追加。

剩余差距已闭合：`recall_history` 工具（2026-09-14）给了模型一个历史读出口——从自然语言问句
（"我之前的住址是什么"）路由到属性，返回按时间排序的版本时间线，项目存储与共享用户存储都查。
配套地，两侧的 slot 剥离清单加入了"之前的 / 上一次的 / 上次的"，让历史问句与事实路由到同一个键。
跨语言 parity 测试（`write-layer outcome matches igm/store.py`）的对照量随之改为当前投影，
`reports/slot-ood-baseline.json` 已按新语义重生成。
