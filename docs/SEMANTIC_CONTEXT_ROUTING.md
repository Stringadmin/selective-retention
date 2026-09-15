# 语义情境路由：RAG 如何接上结果反馈记忆

> 状态：实验原型。验证的是 BGE 检索、情境 ID 与 `OutcomeMemory` 的组合行为；不是端到端 Agent 基准，也没有训练任何模型参数。

## 问题与架构

`OutcomeMemory` 的证据键是稳定的 `(context_id, action)`。真实任务不会直接给出稳定 ID，而会给出新措辞。若把每条新措辞当 ID，已有经验无法复用；若错误地映射到另一个 ID，又会读取不相干的经验。因此 RAG 和经验记忆应分工：

```text
新的任务措辞
  -> BGE 语义检索：候选 canonical context_id
  -> 路由校验/澄清/人工回退（不能只信 top-1）
  -> OutcomeMemory：(context_id, action) 的时间衰减结果证据
  -> 只为已接受的路由推荐动作
```

RAG 在这里不与记忆层竞争：它解决“这次任务像哪个情境”，而 `OutcomeMemory` 解决“该情境下哪些已执行动作最近有可信结果”。错误路由会把正确的经验用在错误的情境上，因此它是整个架构的关键失败面。

## 实验设计

实现：`memory_arch/context_router.py` 与 `memory_arch/run_semantic_transfer.py`。

- 使用本地真实 `BGE-small-zh-v1.5`；8 个运维类 canonical context，每个仅有 **1 条** router prototype。
- 每个 context 在结果记忆中预置两种动作各 8 条已验证的模拟执行结果；500 个随机 seed 仅随机化哪种动作更优和结果噪声。
- 每个 probe set 都有 32 条未进入 router 索引的中文任务措辞。`OutcomeMemory` 的动作效果是模拟结果证据的行为，不是 LLM 或工具调用成功率。
- `calibration` 是最初的 32 条探针。观察到其中的错误多为很小的 top-1/top-2 相似度差后，**事后**固定 `min_margin=0.02`。
- `evaluation` 是另一组 32 条、未参与阈值选择的手工探针。它由实验作者在校准结果已经可见后编写，因而不是独立采样的公开基准，仍可能有作者偏差。

“拒答”表示不自动选择 context 或执行记忆推荐，必须转交给澄清、较强的判别器或人工流程；它不等于把错误任务答对。

## 结果

### 校准集：小间隔恰好捕捉了已观察到的错误

| 指标 | 数值 |
|---|---:|
| BGE top-1 路由 | 30/32 = 93.75% |
| margin >= 0.02 的自动路由覆盖率 | 28/32 = 87.5% |
| 自动接受集合的路由精度 | 28/28 = 100% |
| 被拒答的样本中原本错误的路由 | 2/4 |
| 不做语义路由，直接以新措辞读经验的动作准确率 | 50.45% |
| BGE 路由 + 结果证据的动作准确率 | 96.925% |
| 仅自动接受部分的动作准确率 | 99.914% |

最后三项由预置的模拟结果决定。它们只能说明**在路由正确时**，同一情境的既有结果证据能被新措辞复用；不能说明真实 Agent 有此成功率。

原始记录：[reports/semantic-transfer-bge.json](../reports/semantic-transfer-bge.json)。

### 留出集：margin 不是安全保证

将没有再调过的 `0.02` 用于 `evaluation` 后，结果明显变差：

| 指标 | 数值 |
|---|---:|
| BGE top-1 路由 | 24/32 = 75.0% |
| 自动路由覆盖率 | 24/32 = 75.0% |
| 自动接受集合的路由精度 | 22/24 = 91.67% |
| 8 个错误中被拒答 | 6 |
| BGE 路由 + 结果证据的动作准确率 | 87.669% |
| 仅自动接受部分的动作准确率 | 95.983% |
| 所有查询中“自动且动作正确”的比例 | 71.988% |

两个错误仍被高置信度接受：

| 措辞 | 应有情境 | BGE top-1 | margin |
|---|---|---|---:|
| `配置文件误传了云服务访问凭证。` | `secret_exposure` | `release_failure` | 0.08317 |
| `高峰期页面打开缓慢且请求堆积。` | `latency_incident` | `cache_inconsistency` | 0.02737 |

留出集阈值扫描进一步说明这个问题：

| margin 下限 | 自动覆盖率 | 接受集合精度 | 自动接受错误数 |
|---:|---:|---:|---:|
| 0 | 100.0% | 75.0% | 8 |
| 0.01 | 87.5% | 85.7% | 4 |
| 0.02 | 75.0% | 91.7% | 2 |
| 0.03 | 71.9% | 95.7% | 1 |
| 0.05 | 59.4% | 94.7% | 1 |
| 0.08 | 43.8% | 92.9% | 1 |
| 0.10 | 34.4% | 100.0% | 0 |

`0.10` 是在看过同一留出集后才会选出的数，绝不能作为独立证据或生产阈值；即使在这个集合上，它也牺牲了近 2/3 自动覆盖率。原始记录：[reports/semantic-transfer-bge-evaluation.json](../reports/semantic-transfer-bge-evaluation.json)。

## 得到的结论

本实验支持一个有限结论：**语义检索可以让新措辞复用按结果维护的外部经验；一条 prototype 的 BGE top-1 路由并不可靠，top-1/top-2 的间隔也不足以单独充当安全闸门。**

它不支持“记忆比 RAG 强”或“Agent 已学会自主训练”。在这里，结果记忆的所有优势都依赖 RAG 首先给出正确情境；路由失败时，经验层会放大而不是修复错误的关联。

## 后续架构与验证门槛

在接入插件或自动执行流程之前，需要先完成以下验证，而不是继续调这 32 条文本的阈值：

1. 用多个经过来源审核的 exemplar 为每个 canonical context 建索引，并将“多个 prototype 的检索”与单 prototype、无路由、oracle route 做消融。
2. 为高风险情境增加独立的约束/分类器或澄清回合；回退决策必须衡量误放行率、人工负担和端到端任务成功率。
3. 使用预先冻结、来自真实工具任务或公开基准的任务集；阈值只在开发集确定一次，在测试集只运行一次。
4. 让结果写入来自实际的可验证信号（测试、部署、人工审核），报告错误路由写入/读取的污染率、p95 延迟、持久化字节和恢复速度。

在满足这些门槛前，`ContextRouter` 与 `OutcomeMemory` 只应作为研究原型，不能用于无监督的生产决策。

## 复现

在已下载 BGE 模型的 WSL 环境：

```bash
cd /mnt/f/FIP-Transformer

# 阈值是在这组校准探针上事后观察得到的
/home/omnichat/fip-venv/bin/python -m memory_arch.run_semantic_transfer \
  --seeds 500 --min-margin 0.02 --probe-set calibration \
  --output reports/semantic-transfer-bge.json

# 固定同一阈值，运行另一组手工措辞；不要据此反向调参
/home/omnichat/fip-venv/bin/python -m memory_arch.run_semantic_transfer \
  --seeds 500 --min-margin 0.02 --probe-set evaluation \
  --output reports/semantic-transfer-bge-evaluation.json
```
