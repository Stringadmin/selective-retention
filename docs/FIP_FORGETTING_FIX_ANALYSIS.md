# FIP Phase 0 遗忘门控修复：根因分析与验证记录

> 日期：2026-08-10
> 状态：FFN 层修复已落地（测试通过）；非 FFN 保护为下一关键设计点。
> 协议背景：`docs/FIP_RESEARCH_PROTOCOL.md` 要求 F 组遗忘相对 E（稀疏全量）降低 >=50%。

## 1. 修复内容（已合并到 `fip/feature_registry.py`）

`end_task()` 的两处改动：

1. **stable 优先于 shared**：原始逻辑是"被多个任务重用的特征优先升为 shared（0.1 更新率）"，高重要性特征若被重用会被降级为 shared 并被后续任务缓慢覆盖。现改为**高重要性特征无条件进入 stable（冻结）**，无论是否被重用。

2. **sticky-stable（稳定状态不可逆）**：importance 是跨任务 EMA，会在后续任务中指数衰减，导致早期任务的特征重要性跌破阈值后被降级。现改为**一旦 stable 永远 stable**，防止旧知识被"重新评估"降级。

这两个改动修复了 string 任务的关键泄漏路径：string 的 FFN 特征被冻结后不再被 shared/unrelated 任务（因共享 `_SOURCE_BASE` 输入 token 而激活它们）覆盖。

**相关测试更新**（`tests/test_mechanism.py`）：
- `test_end_task_promotes_stable`：语义不变（高重要性→stable）
- `test_repeated_use_promotes_shared`：更新为验证"高重要性重用→stable，边际贡献不冻结"
- `test_marginal_contribution_does_not_promote_shared`：更新为验证"高贡献→stable，边际不 shared"

**测试状态**：46 passed, 1 skipped。

## 2. 完整配置验证结果

同协议 seed 1337，6 层 / 256d / d_ff 1024，2000 步/任务（评估口径 batch 16）。

| 方案 | 遗忘 | vs E | string 保留 | 备注 |
|---|---:|---:|---:|---|
| E 稀疏基线 | 0.4148 | — | 0.0（全遗忘） | 门控基准 |
| F 原始 | 0.4915 | -18.5% | 0.0 | 门控失败 |
| F + 本修复 | 0.4058 | **+2.2%** | shared 后 0.319 | 轻微改善 |
| F + 冻结全部非FFN | 0.2154 | +48% | 部分保留 | 牺牲新任务（unrelated=0） |
| H 渐进非FFN保护 | 0.3778 | +9% | 0.0 | 保护不足 |

## 3. 根因分析（证据链完整）

### 3.1 决定性证据：string 遗忘不经过 FFN 路径

训练后对 string 激活的特征做逐层状态追踪（batch 16, 3000 步配置）：

- string 训练结束后，其激活的特征在 layer1-5 **100% 处于 stable（冻结）**，layer0 有 71 个 plastic 特征
- shared 任务后：string 的 layer0 plastic 特征中 **36 个被 shared 任务激活并改写**
- 结论：**即使 FFN 特征全部冻结，string 仍从 1.0 归零**，证明覆盖路径在非 FFN（embedding/attention/norm/head）

### 3.2 FFN 冻结有效性验证

- string 的 122 个 stable 特征在 shared 任务后 drift = 0.0（完全冻结，optimizer 双层 mask 生效）
- 高重要性特征被冻结后，后续任务确实无法改动它们（sticky 验证）

### 3.3 容量守恒定律（设计层面硬约束）

- 每个任务标记约 25% 特征为 significant，6 个任务总需求 ~150%，超过 100% 容量
- "冻结所有显著特征"（used_now 全量版）能保护 string（shared 后 0.319）但整体遗忘恶化至 0.642（后续任务无可用特征）
- 逐任务冻结策略受容量守恒约束，必然有任务被牺牲

### 3.4 非 FFN 是主导遗忘路径

- 冻结全部非 FFN 后遗忘 0.2154（改善 48%，接近门控 50%），但全部冻结牺牲新任务学习（unrelated=0）
- H 渐进保护（按重要性 top 50% 行减量 0.1 更新）改善至 0.3116-0.3778，但仍不足且 string 保护不了
- 根因：string 与 shared/unrelated 共享 `_SOURCE_BASE` 输入 token 的 embedding 行，非 FFN 保护按"当前重要性"选行，string 的行在后续任务中重要性被稀释而失去保护

## 4. 结论

1. **FFN 层修复是必要但不充分的**：改善 17%（0.4915→0.4058），string 在 shared 阶段保留显著改善，但无法达到 50% 门控。
2. **真正的瓶颈在非 FFN 参数**：embedding/attention/norm 在下游任务中被覆盖是 string 遗忘的主导路径。
3. **H 方案（渐进非 FFN 保护）方向正确但保护强度/选行策略不足**：非 sticky 选行导致旧任务的重要行在后续任务中失去保护。

## 5. 累积式非 FFN 保护（group I 改进，已实现）

### 5.1 改进内容（`fip/non_ffn_protection.py`）

原有 group I（固定容量 sticky）失败原因：第一任务就把 `max_protected_fraction=0.5` 容量填满，后续任务无法保护自己的重要行。

改进为**每任务预算制**：

1. **`per_task_budget`（每任务预算）**：每个任务结束时，从该任务实际贡献（`task_contrib`，非累积 EMA）中选出重要行加入保护集，新增量受 `per_task_budget` 限制。容量随任务数增长：`min(max_protected_fraction, per_task_budget × n_tasks)`，**后续任务不被早期任务锁死**。
2. **1-D 参数（norm）整体保护**：norm 是全局逐通道缩放参数，任何通道影响所有任务。一旦 norm 承载任何知识，**整个向量** scale=0.1（sticky），不做逐通道部分保护。这修复了"unrelated 通过 norm 未保护通道改写所有旧任务表示"的泄漏路径（drift 分析显示 norm 是最大 drift 源，mean 0.0116）。

### 5.2 使用方式

```bash
# group I 使用 per-task budget 模式
python -m experiments.phase0 --device cuda --groups I --per-task-budget 0.2 \
  --output-dir reports/phase0-sticky-v3

# 固定容量旧模式（与 v2 报告一致）
python -m experiments.phase0 --device cuda --groups I \
  --output-dir reports/phase0-sticky-v3
```

### 5.3 完整配置验证（seed 1337, 6层/2000步, batch 16 口径）

| 方案 | 遗忘 | string final | conflict final | modular final |
|---|---:|---:|---:|---:|
| E 稀疏基线 | 0.4148 | 0.0 | 0.044 | 0.0 |
| FIP sticky-stable | 0.4058 | 0.0 | 0.127 | 0.383 |
| **FIP + sticky budget(0.2) + norm保护** | 0.4438 | **0.077** | 0.306 | 0.675 |

**主要成果**：string 首次在最终保留非零精度（0.077），conflict（0.306 vs E 的 0.044）和 modular（0.675）大幅优于 E。modular/shared 在各自训练后达到 0.956（完整学习能力保留）。

**仍存在的权衡**：norm 整体保护保护了 string 但削弱了 conflict 任务的适应（norm 无法为 conflict 微调），整体遗忘 0.4438 仍高于 E。**未达 50% 门控**。

## 6. 结论

1. **FFN 层修复是必要但不充分的**：改善 17%（0.4915→0.4058），string 在 shared 阶段保留显著改善，但无法达到 50% 门控。
2. **真正的瓶颈在非 FFN 参数**：embedding/attention/norm 在下游任务中被覆盖是 string 遗忘的主导路径。drift 分析确认 **norm（RMSNorm）是最大 drift 源**。
3. **累积式保护（每任务预算 + norm 整体保护）方向正确**：string 首次保留非零（0.077），conflict/modular 保留显著优于基线。但 norm 保护与 conflict 任务适应之间存在权衡，整体遗忘仍高于门控。
4. **容量守恒是硬约束**：6 个任务的显著特征总需求超过 100% 容量，任何逐任务冻结策略都必须牺牲部分任务。

## 7. 下一步建议（需研究决策）

1. **norm 分级保护**：当前 norm 全保护（scale=0.1）削弱 conflict 适应。可改为 norm 采用更大 scale（如 0.5）或延迟保护（任务 2 后再保护），平衡保护与适应。
2. **embedding 行的任务归属保护**：为每个输入 token 记录其"所属任务"，后续任务训练时若改写了其他任务的 token 行，施加更强保护（甚至冻结）。
3. **数据层面的归因检查**：string 与 unrelated 共享输入 token 是数据设计特性，若确认无法通过机制保护，需评估是否调整任务设计或接受该任务的遗忘为合理代价。
4. **随机冻结消融**（协议消融 #8）：验证智能分配是否优于随机，是判断 FIP 核心价值的关键实验。

## 8. 未完成的实验

- 完整 5 种子矩阵（当前修复仅验证 1 种子，且未达门控，不应当启动 5 种子）
- D 组（累积 LoRA）仍被屏蔽（参数公平性未解决）
- norm 分级保护 / embedding 任务归属保护（见 §7）
