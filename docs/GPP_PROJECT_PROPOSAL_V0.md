# 新课题规划：GPP — Graded Parameter Protection（分级参数保护）

> 日期：2026-08-10
> 状态：课题提案 v0.1（待评审）
> 来源：FIP Phase 0 攻坚的实证结论——FFN 特征隔离无效，而非 FFN 逐行参数保护是唯一实质有效机制（+13.2% 遗忘改善）。工程资产全部复用。

## 0. 一句话定位

**在优化器层面做逐行分级参数保护**：按参数行的重要性（梯度 EMA）决定其更新缩放（0 / 0.1 / 1.0），累积式、不可逆地保护已学任务的关键参数行，解决持续学习中的灾难性遗忘——不依赖回放、推理任务 ID 或额外的正则化损失。

## 1. 为什么值得做（与现有方法的差异点）

| 对比对象 | 它们的做法 | GPP 的差异 |
|---|---|---|
| **EWC**（soft 正则化） | 在 loss 加 `Σ Fisher×(θ-θ*)²` 惩罚项 | GPP 在**优化器 delta 层投影**，直接控制实际更新。且 EWC 的 Fisher 是全局对角、逐任务叠加，不设容量；GPP 逐行 EMA + 每任务预算 |
| **硬冻结/掩码**（PackNet 等） | 二元冻结，牺牲可塑性 | GPP 三级缩放（0/0.1/1.0），保留共享能力微调 |
| **LoRA/专家隔离** | 每任务加参数，参数量增长 | GPP 固定参数量，无任务 ID |
| **记忆重放** | 需要旧数据 | GPP 无回放 |

**关键工程洞察（已在 FIP 中验证）**：仅靠梯度 hook 无法冻结参数——AdamW 的解耦权重衰减和旧动量在梯度为 0 时仍会移动参数。必须**在优化器 step 之后对实际 delta 做投影**。这是 GPP 的技术底座，也是 EWC 类方法没有解决的真实问题。

## 2. 方法定义（M1 → M3 渐进复杂度）

### M1：逐行重要性 + 每任务预算累积保护（核心）
- 对每个非 FFN 参数（embedding 行 / attention 行 / norm 向量）维护：
  - `importance`：EMA(|grad| per row)，跨任务累积
  - `task_contrib`：当前任务内的贡献（选择"该任务真正用到的行"）
- `end_task` 时：从 `task_contrib` 的 top-quantile 行中，按**每任务预算**（如 20%）加入保护集；容量随任务数增长 `min(cap, budget×n_tasks)`；**已有保护行永不释放**（sticky）
- 被保护行更新缩放 = `protected_scale`（默认 0.1）；未保护 = 1.0；专属冻结 = 0.0
- 1-D 参数（norm）整体保护：norm 是全局缩放器，任何通道影响所有任务，一旦承载知识全向量 scale=0.1

### M2：token 归属冻结（可选增强）
- 记录每个 token 的任务归属；仅被单一任务使用的 token 行 scale=0（完全冻结）
- **注意**：在 token 高度共享的任务套件下收益有限（FIP 已证），设计新任务数据时保留此模块作消融

### M3：FFN 特征隔离（可选，作为消融而非核心）
- 若未来想保留 FIP 的"智能分配"卖点（25.7% vs 随机），可叠加 FFN 层，但**不作为 GPP 主干**

## 3. 可证伪的核心假设

**H1（保护有效性）**：GPP 的遗忘 ≤ 全量微调的一半，且新任务学习 ≥ 全量微调的 90%。
**H2（累积优于重选）**：累积式（sticky）保护优于每任务重选（non-sticky），turnover 更低。
**H3（预算优于固定帽）**：容量随任务增长优于固定容量帽（后者被第一任务填满）。
**H4（优化器投影必要）**：移除优化器层投影、仅用梯度 hook 时，冻结参数仍会漂移（机制必要性证明）。
**H5（任务归属价值）**：token 归属冻结在 token 重叠度高的任务流中提供额外保护。

## 4. 实验设计

### 4.1 阶段 0：机制门控（复用 FIP 基建，~2 周）
- 复用：`NonFFNProtector`、`FeatureMaskedAdamW`、`phase0.py` runner、公平性协议
- 任务套件：**保留 FIP 的 6 任务合成套件**（关联记忆/冲突更新/组合规则/共享规则/无关任务），数据生成器 `data/synthetic_tasks.py` 不变——结果可直接与 FIP 的 0.3952 对照
- 门控：
  - H1: 遗忘 ≤ E 的 50%，新任务 ≥ 90%
  - H4: 冻结行在"仅梯度 hook"下 drift > 1e-4，在"优化器投影"下 < 1e-7
  - H2/H3: sticky vs non-sticky、budget vs fixed-cap 的消融对比
- 通过后跑 5 种子矩阵

### 4.2 阶段 1：标准 CL 基准（~4 周）
- 数据：Split CIFAR-100 / Split MNIST（图像）或 5-domain NL 文本（可选，若有资源）
- 基线矩阵：Dense FT、EWC、PackNet-style 硬冻结、LoRA、Replay(5%)、GPP
- 指标：Average Accuracy、Average Forgetting、BWT/FWT、参数量公平（GPP 与 Dense 等参数量）

### 4.3 阶段 2：放大（~4 周）
- 50M 参数模型 + 自然语言任务流（对话→代码→数学→生物→法律）
- 报告：遗忘、新任务能力、实际被保护行比例、推理延迟增量

## 5. 里程碑

| 里程碑 | 内容 | 判定 |
|---|---|---|
| M0（本周） | 机制门控 + H1/H4 验证 | 遗忘 ≤ E×0.5 且新任务 ≥90% |
| M1（+2周） | 5 种子 + 消融（H2/H3/H5） | 3 种子稳定 |
| M2（+4周） | 标准基准（CIFAR/文本） | 对 EWC/冻结/LoRA 全面超越 |
| M3（+8周） | 50M NL 放大 + 论文初稿 | 门控指标 + 可复现 |

## 6. 风险与备选

| 风险 | 应对 |
|---|---|
| GPP 在合成任务上达标但标准基准上失效 | 优先做 H4（机制必要性）和 5 种子，确认机制根基；阶段 1 用双基准交叉验证 |
| norm 保护 vs 新任务适应的权衡（FIP 已见） | 调 `protected_scale`（0.1→0.5）或延迟保护（任务 2 后），纳入 M0 调参 |
| 与 EWC 被审稿人视为相似 | 论文重点突出"优化器层 delta 投影"与"每任务预算累积"两个独特点，H4 实验作为差异证明 |
| 保护行累积导致可塑性枯竭 | 容量上限 + `protected_scale>0`（非全冻结），报告保护行比例随任务变化 |

## 7. 与 FIP 的切割关系

- **保留复用**：`NonFFNProtector`、`FeatureMaskedAdamW`、`phase0.py`、公平性协议、合成任务套件
- **弃用**：FFN 特征状态机（`feature_registry.py` 的 stable/shared/plastic/free）、Top-K 稀疏、RollbackManager（作为诊断工具保留）
- **核心新卖点**：优化器层 delta 投影 + 逐行 EMA 重要性 + 每任务预算累积

## 8. 命名建议

- GPP: Graded Parameter Protection
- 备选：CIP (Cumulative Importance Projection)、GDP (Graded Delta Projection)
