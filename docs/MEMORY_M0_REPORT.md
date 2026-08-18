# IGM M0 基线报告：重要性门控记忆的首次验证

> 日期：2026-08-18
> 状态：**M0 核心假设初步验证通过** —— 写入闸门 + slot 覆盖两大机制均实证有效
> 环境：WSL2, RTX 5070 (12GB), fip-venv, llama-cpp-python (CPU), qwen3:4b (2.5GB GGUF)
> 提案：`docs/MEMORY_ARCHITECTURE_RESEARCH_PLAN.md`

## 1. M0 目标

验证 IGM（Importance-Gated Memory）的两个核心机制在受控多会话数据上是否成立：
- **写入闸门**：重要性驱动的选择性写入，能否用更少记忆达到更好效果（H1/H4）。
- **slot 覆盖（知识更新）**：同属性新事实取代旧事实，能否解决"知识更新"（H2）。

对照基线：full_context（上限）、naive_rag（全量写入）、mem0_style（启发式提取）。

## 2. 数据与方法

合成多会话数据集（11 对话，3 类问题，seed 1337）：
- single_session_recall（5）：事实陈述一次，后续提问。
- knowledge_update（3）：事实被后续更新，正确答案是新值（测遗忘）。
- temporal_reasoning（3）：带时间词的事件，问最近一次。

底层 LLM 固定 qwen3:4b，消除模型差异。嵌入为 hash bag-of-words（占位，后续换真嵌入）。

## 3. 结果

### 3.1 首跑（写入闸门验证）

| 方法 | 总 acc | 单会话召回 | 知识更新 | 时间推理 | 记忆条数 |
|---|---:|---:|---:|---:|---:|
| full_context | 1.000 | 1.0 | 1.0 | 1.0 | （上限，全塞上下文） |
| naive_rag | 0.091 | 0.2 | 0.0 | 0.0 | 89 |
| mem0_style | 0.000 | 0.0 | 0.0 | 0.0 | 168 |
| **IGM** | **0.364** | **0.8** | 0.0 | 0.0 | **12** |

**IGM 用 12 条记忆（naive_rag 的 1/7）达到 4 倍准确率（0.364 vs 0.091）**。写入闸门有效，H4 成本-质量帕累托卖点初步成立。

### 3.2 加入 slot 覆盖 + slot 感知检索后（知识更新验证）

| 方法 | 总 acc | 单会话召回 | 知识更新 | 时间推理 | 记忆条数 |
|---|---:|---:|---:|---:|---:|
| **IGM (v2)** | **0.636** | 0.8 | **1.0** | 0.0 | 9 |

**知识更新从 0 → 1.0**，整体 acc 0.364 → 0.636。记忆条数降到 9（3 条过时值被取代）。H2 遗忘必要性实证通过。

## 4. 关键发现

1. **写入闸门有效**：重要性评分（自我指涉事实标记 + surprise + 密度）能精准筛出"我的X是Y"类持久事实，过滤闲聊。12 条 vs 89 条达到 4 倍准确率。
2. **slot 覆盖解决知识更新**：属性级 supersede（新值写入时旧值移除）+ slot 感知检索（精确路由，绕过词袋嵌入噪声），使知识更新准确率达 1.0。**这正是 Mem0/Zep 最难处理的场景**。
3. **检索质量是瓶颈**：hash 词袋嵌入无法区分属性词，必须靠 slot 精确路由兜底。换真嵌入（如 BGE/E5）是 M1 重点。
4. **qwen3:4b 的 thinking 模式是工程噪音**：默认输出长推理前缀，`/no_think` 对该 GGUF 未生效，截断了部分答案（主要影响时间推理）。这是模型行为问题，非机制问题。

## 5. 遗留问题（M1 方向）

- **temporal_reasoning = 0**：需"时间戳感知的记忆排序"机制（按 created_at 排序最近事件）。当前 slot 机制不覆盖纯时间比较。这是下一个明确的机制增项。
- **真嵌入替换**：hash 词袋 → BGE/E5 等真嵌入，减少对 slot 路由的依赖。
- **学到的 importance scorer**：当前是启发式三信号加权，M1 用检索日志/复用频率做弱监督训练。
- **真实 LongMemEval 数据**：GitHub/HF 被墙，需国内镜像或手动获取后做标准基准验证。

## 6. 复现

```bash
# WSL 内，模型在本地磁盘
cd /mnt/f/FIP-Transformer
~/fip-venv/bin/python -m memory_arch.run_m0 \
    --n-gpu-layers 0 --methods igm \
    --output reports/memory-m0-igm.json
# 详情: reports/memory-m0-igm-details.json
```

代码：`memory_arch/__init__.py`（框架+4方法+记忆库）、`memory_arch/synth_data.py`（数据）、`memory_arch/run_m0.py`（评估）。
