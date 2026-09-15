# LongMemEval-S：角色感知的 Session 证据检索

> 状态：离线检索实验。它只衡量标注的证据 session 是否出现在 top-k，不衡量生成答案是否正确；没有调用 LLM、没有写模型参数，也不构成“IGM 优于 RAG”或 AGI 的证据。
>
> **更正（2026-09-14）：** 本页所有数字都用中文 BGE 编码英文数据。换成 `bge-small-en-v1.5` 在相同 test 分区复跑后：整体 `recall_all@10` 从 **86.91% → 93.98%**，而 k=10 上 user-only 与 role-aware 双双饱和到 93.98%——本页的核心结论 "+0.52pp" 因此看着归零，实际是天花板效应使 k=10 不再有区分度。角色路由的真实收益在**排序**上：整体 `recall_all@1` 27.75%→28.80%、`nDCG@5` 88.87%→89.75%，目标题型 `single-session-assistant` 的 `recall_any@1` 86.05%→95.35%。此外本页的召回指标看不到 top-k 内部的顺序问题：knowledge-update 题里陈旧值排在新值之前的比例达 61.1%（top-10 内）/ 63.9%（top-1）。两处修正与后续实验见[陈旧值泄漏、写入层覆盖度与 reader 评测](LONGMEMEVAL_STALE_LEAK_AND_READER.md)。

## 为什么做这个实验

此前的 Versioned IGM 与 OutcomeMemory 只验证了受控的更新、审计和反馈语义。要研究读侧，首先需要在含大量历史会话的公开数据上确认一个更基本的问题：检索器是否能把回答所需的会话带回上下文。

LongMemEval-S 的标注提供 `answer_session_ids`。本实验不让模型作答，只把每个历史 session 当作一个文档，计算标注 session 在 top-k 中的比例。这样把“证据没有取到”和“reader 没有用好证据”严格分开。

## 数据与协议

- 数据：`F:/benchmarks/longmemeval-cleaned/longmemeval_s_cleaned.json`，500 个实例；30 个 abstention 问题与官方检索评估一样跳过，实际计分 470 个实例。
- 数据 SHA-256：`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`。
- 索引粒度：一个有时间戳的 history session 为一个文档。user-only 投影仍保留 assistant-only session 的 ID，以匹配官方扁平 session 索引；其文本为空。
- 嵌入器：本地 `BAAI/bge-small-zh-v1.5`。它是中文模型而数据主要为英文，因此以下数值只是此本地配置的可复现检索基线，不能与 LongMemEval 论文或排行榜直接比较。
- 指标：官方风格的 `recall_any@k`、`recall_all@k`、nDCG@k。`recall_all` 对多证据题要求所有标注 session 都进入 top-k。
- 分区：用 `sha256(question_id)[0] % 5 == 0` 划 20% 内部开发分区（88 个计分实例），其余为测试分区（382 个）。LongMemEval-S 公开发布时只有一个评估集，因此这不是官方 train/test split，也不是新的 leaderboard 成绩。

全量数据的固定基线在设计本角色路由前已经运行过。因此即使路由词表只从开发分区构造，测试分区也不能被描述为完全盲测或独立发表级测试；它的作用是限制本次规则的直接过拟合，而不是提供强泛化保证。

实现：[memory_arch/longmemeval_retrieval.py](../memory_arch/longmemeval_retrieval.py)。原始报告均位于 `reports/`。

## 固定基线：写入角色会影响读侧检索

在全量 470 个可评分实例上，k=10 的结果如下：

| 索引文本 | recall_any@10 | recall_all@10 | nDCG@10 |
|---|---:|---:|---:|
| all-turns | 87.45% | 72.55% | 68.31% |
| user-only | 92.34% | 85.96% | 77.54% |

原始报告：[all-turns](../reports/longmemeval-s-bge-session-all.json)、[user-only](../reports/longmemeval-s-bge-session-user-only.json)。

user-only 对多数用户事实和 knowledge-update 问题更好：例如 knowledge-update 的 `recall_all@10` 为 94.4%，all-turns 为 65.3%。但直接删去所有助手文字会伤害 `single-session-assistant`。因此“只写用户消息”不是一般的长期记忆解法；角色是读写策略的一部分。

## 角色感知读侧

对每个问题，路由器在**不读取** `question_type` 或答案标注的条件下选择索引：

```text
问题文本
  -> 明确在问助手之前的回答？ -> all-turns session index
  -> 否则                         -> user-only session index
```

词表只根据开发分区的助手历史措辞冻结：`previous chat/conversation`、`last time`、`you recommended/told me/mentioned/said/created`、`did you say/tell/mention/recommend`、`we outlined/discussed`。代码中的 `needs_assistant_history()` 是大小写无关的确定性正则，不是训练出的分类器。普通的 `you` 不触发路由。

选择 `recall_all@10` 作为首要完整证据指标。开发分区的比较为：

| 索引策略 | recall_all@10 | nDCG@10 |
|---|---:|---:|
| all-turns | 68.18% | 67.42% |
| user-only | 81.82% | 75.79% |
| role-aware | 81.82% | 76.13% |

role-aware 没有提高开发集的完整召回，只提升了助手历史题的排序；它没有损害 user-only 的整体优势。基于这一点冻结词表，未再根据测试结果更改。

测试分区结果：

| 索引策略 | recall_any@10 | recall_all@10 | nDCG@10 |
|---|---:|---:|---:|
| all-turns | 87.70% | 73.56% | 68.51% |
| user-only | 92.15% | 86.91% | 77.94% |
| role-aware | 92.67% | 87.43% | 78.60% |

测试集 all-turns 数值由全量固定基线和开发分区均值按实例数反推；其余两行直接来自对应的测试分区报告。原始报告：[all-turns full](../reports/longmemeval-s-bge-session-all.json)、[all-turns development](../reports/longmemeval-s-bge-session-all-development.json)、[user-only test](../reports/longmemeval-s-bge-user-test-selected.json)、[role-aware development](../reports/longmemeval-s-bge-role-aware-development.json)、[role-aware test](../reports/longmemeval-s-bge-role-aware-test.json)。

role-aware 相对 user-only 的全部 `recall_all@10` 只增加 **0.52 个百分点**。收益集中在 `single-session-assistant`：`88.37%` 对 `83.72%`（+4.65 个百分点），nDCG@10 为 `86.35%` 对 `79.84%`（+6.51 个百分点）。其它题型的完整召回没有变化；temporal-reasoning 的 nDCG@10 反而低 0.29 个百分点。这是针对特定证据角色的微小检索收益，不是通用记忆能力的跃升。

## 失败分析

在测试分区，用 benchmark 的 `single-session-assistant` 标签仅作事后诊断，文本路由有：

| 路由诊断 | 数量 |
|---|---:|
| 真正例 | 37 |
| 假负例 | 6 |
| 假正例 | 1 |
| 真负例 | 338 |
| 精确率 | 97.37% |
| 召回率 | 86.05% |

漏判包括“you provided me earlier”、省略了“previous conversation”的“餐厅/电话是什么”，以及“previous chess game”。唯一误判是时间推理题中的 `last time`，它指的是用户事件时间而非助手回复。用测试集再加词会使同一份结果失去验证意义，因此没有这样做。

这说明只靠措辞识别助手历史不足以承担生产路由：它会漏掉自然表达，并可能把时间短语误读为会话回忆。后续应在独立数据上比较来源/角色元数据、训练但经校准的意图分类器、澄清回合和更强的混合检索，同时报告误路由的下游回答、权限和隐私影响。

## 结论与边界

本实验支持一个有限结论：**在这个本地 BGE 配置下，分开 user-only 与 all-turns 索引，并对明确的助手历史查询选择后者，能保住 user-only 的大部分收益，并小幅改善助手生成内容的 session 证据排序。**

它不支持以下说法：

- IGM、Versioned IGM 或 OutcomeMemory 的总体效果优于 RAG；这里实现本身就是一种 RAG 读侧策略。
- 系统已“自主学习”或更新了模型权重；本实验没有训练、反馈写入或在线优化。
- session evidence recall 等于 LongMemEval 的官方问答准确率、真实 Agent 成功率或用户满意度。
- 这组正则适合生产；它只有内部开发/测试诊断，且公开数据的全量基线已先被查看。

下一次应优先做端到端且预注册的研究：冻结索引策略和 prompt，分别测无记忆、all-turns RAG、user-only RAG、带来源元数据的 role-aware RAG；以官方 QA/任务成功率、证据召回、prompt token、持久化字节、p95 延迟及错误泄漏率共同报告。只有在独立样本上稳定时，才讨论将角色语义并入 IGM/Versioned IGM 的写入与读出接口。

## 端到端 QA 的执行门槛（2026-08-24）

本轮**没有**新增 LongMemEval QA 正确率。官方 `evaluate_qa.py` 需要一个
LLM-as-judge（公开脚本只配置了 GPT-4o / GPT-4o-mini 或本地服务在
`localhost:8001` 的 Llama-3.1-70B），当前环境没有运行中的兼容裁判服务，
也没有配置外部 API 作为替代。因此不能用字符串包含、答案标签或 retrieval
recall 冒充官方 QA 评测。

本地 Qwen3:4B GGUF 也未通过 reader 预检：以 `n_ctx=8192` 和
`n_gpu_layers=99` 处理第一个 oracle evidence prompt（约 18,042 个字符）时，
进程在 CPU 上运行 9 分 25 秒仍未给出首个结果，随后主动停止；其默认长思考
输出也会在较短生成预算中截断最终答案。另一个本地 `gpt-oss-20b` 权重缺少
MXFP4 所需的 `kernels` 运行时，Transformers 会退回 BF16 反量化，在 12 GB
显存设备上不能作为已验证的批量 reader。上述都是环境/运行门槛，不是任何
检索方法的失败分数。

因此当前结论保持为 session 证据检索，不再把角色路由的 `+0.52` 个百分点
外推为生成质量。若后续获得一个可稳定运行、可记录模型版本与 prompt 的
reader，以及独立或经过校准的 judge，应先在预先冻结的 20 条 oracle evidence
健康检查上报告生成耗时、可完成率与 judge 一致性；只有该门槛通过后，才运行
all-turns、user-only 和 role-aware 三组完整 QA 对照。测试集不应再被用来
调整角色路由词表。

## 复现

在包含本地 BGE 模型的 WSL 环境中：

```bash
cd /mnt/f/FIP-Transformer
MODEL=/home/omnichat/.cache/modelscope/models/BAAI--bge-small-zh-v1.5/snapshots/master
DATA=/mnt/f/benchmarks/longmemeval-cleaned/longmemeval_s_cleaned.json

# 固定 user-only 基线
/home/omnichat/fip-venv/bin/python -m memory_arch.longmemeval_retrieval \
  --input "$DATA" --output reports/longmemeval-s-bge-user-test-selected.json \
  --embed-model "$MODEL" --partition test --user-only

# 开发集已冻结的角色路由；不要用测试结果改词表后再运行
/home/omnichat/fip-venv/bin/python -m memory_arch.longmemeval_retrieval \
  --input "$DATA" --output reports/longmemeval-s-bge-role-aware-test.json \
  --embed-model "$MODEL" --partition test --role-aware
```
