# Selective Retention

关于"重要性驱动的选择性保留"的研究代码和实验：从持续学习的参数保护，到 Agent 记忆的写入层。

**配套长文**：[《我试图让 AI 学会"不遗忘"，撞了三次墙后搞明白的三件事》](docs/ARTICLE_DRAFT.md)

## 结论（TL;DR）

三个都能复现的发现：

1. **梯度 hook 冻不住参数。** AdamW 的解耦权重衰减和历史动量在梯度为零时照样移动参数（实测 drift 9.94e-5）。要在优化器层对实际 delta 做投影才能真冻住（drift = 0）。
2. **同一套"重要性筛选 → 执行层强制保留/淘汰"的结构，参数空间（FIP/GPP）和信息空间（IGM）都适用。**
3. **RAG 管读不管写。** 标准 RAG（强嵌入 + 全量存储）在矛盾解决场景准确率 0%；加个写入层（slot 覆盖）到 100%，记忆量还减到 1/7。

## 决定性实验

同一属性被连续更新 5 次（新旧值语义近似），问"当前值"：

| 方法 | 准确率 | 记忆用量 |
|---|---:|---:|
| 标准 RAG（信检索第一名） | 0.00 | 28 条 |
| 标准 RAG + 时间排序 | 0.60 | 28 条 |
| **RAG + IGM 写入层（slot 覆盖）** | **1.00** | **4 条** |

![决定性实验](docs/fig2_decisive_experiment.png)

IGM 不是替代 RAG，是装在 RAG 前面的写入层（重要性闸门 + slot 覆盖），向量库和嵌入都不动：

![RAG vs IGM 写入层架构](docs/fig1_rag_vs_igm.png)

## 复现

```bash
# 决定性实验（不调 LLM，秒级）
python -m memory_arch.run_decisive --embed-backend bge

# 训练写入打分器（权重可解释，held-out ≈ 0.956）
python -m memory_arch.train_scorer

# M0 基线对比（需要本地 LLM）
python -m memory_arch.run_m0 --methods naive_rag igm --embed-backend bge
```

嵌入模型 [BGE-small-zh](https://modelscope.cn/models/BAAI/bge-small-zh-v1.5) 要单独下载（HuggingFace 网络受限用 ModelScope）。本地 LLM 基线用 [qwen3:4b](https://ollama.com/library/qwen3)（GGUF）。

## 作为库用（igm）

`igm/` 是可 pip 安装的零依赖写入层，接进 RAG 系统：

```python
from igm import Memory

mem = Memory()
mem.add("我的住址是北京。")
mem.add("更新一下，我的住址现在是深圳了。")   # slot 覆盖旧值
mem.query_texts("我现在的住址是什么？")         # -> 只有深圳
```

详见 [igm/README.md](igm/README.md)。

## 作为 DeepSeek Harness 插件用（dsh-igm-memory）

同一套写入层做成了 DSH 插件，给 agent 加"闸门 + 覆盖 + 跨会话记忆"：

- `remember_fact`：agent 记事实走 IGM 闸门，同属性新值覆盖旧值
- `recall_fact` + 会话启动注入：新会话开局能看到此前记住的事实
- JSON 持久化 + consolidate 遗忘

```sh
dsh plugin --profile web add ./dsh-igm-memory
dsh web   # 重启加载
# 会话里：记住我的住址是北京 → 改记住是深圳 → 新会话问"我住址是哪" → 答深圳
```

代码在 `dsh-igm-memory/`，详见 [dsh-igm-memory/README.md](dsh-igm-memory/README.md)。

## 仓库结构

```
igm/                    # 可安装的 RAG 写入层库（零依赖核心）
  memory.py             #   Memory 门面：add / query / consolidate
  gate.py               #   WriteGate：重要性闸门 + slot 提取
  store.py              #   MemoryStore：slot 覆盖 + 遗忘生命周期
  embedders.py          #   可插拔嵌入（hash / BGE / 自定义 callable）
  test_igm.py           #   17 个单元测试
dsh-igm-memory/         # DeepSeek Harness 插件（cordis bundle）
  lib/index.js          #   remember_fact / recall_fact / 注入 / 持久化 / 遗忘
  cordis.patch.yml      #   插件注册层
  test 脚本              #   scripts/test_igm_plugin.mjs（15 项）
memory_arch/            # Agent 记忆研究代码（IGM 的实验原型）
  scorer.py             #   可学习的 importance 打分器
  run_decisive.py       #   决定性实验：RAG vs RAG+IGM
  run_m0.py             #   M0 基线评估 runner
fip/                    # 持续学习：FIP / GPP 参数保护
  optimizer.py          #   FeatureMaskedAdamW：优化器层 delta 投影
  non_ffn_protection.py #   NonFFNProtector：逐行重要性保护
experiments/            # 持续学习实验 runner（phase0 等）
examples/               # quickstart.py 等示例
docs/                   # 文章、研究报告、配图（PNG/SVG）
reports/                # 原始实验结果 JSON（可审计）
```

## 文档导航

- **文章**：[docs/ARTICLE_DRAFT.md](docs/ARTICLE_DRAFT.md)
- 记忆架构：[研究方案](docs/MEMORY_ARCHITECTURE_RESEARCH_PLAN.md) · [M0 报告](docs/MEMORY_M0_REPORT.md)
- 持续学习：[FIP 终局决策](docs/FIP_PHASE0_FINAL_DECISION.md) · [GPP 提案](docs/GPP_PROJECT_PROPOSAL_V0.md) · [Split MNIST 报告](docs/SPLIT_MNIST_REPORT.md)

## 边界

- 实验在受控合成数据上做的，记忆基线是 4B 本地模型。方向可信，绝对数字别当真。
- 标准 RAG 够用时别加这层。IGM 的价值只在"全量写入会崩"的场景：大规模、频繁更新、矛盾解决。
- IGM 是 RAG 的增强件，不是替代品。

## License

[MIT](LICENSE)
