# FIP Architecture v0

> 冻结日期：2026-07-29
> 范围：FIP v0 只改 FFN，不重写整个 Transformer。

## 1. 设计目标

知识性特征集中在 FFN，且最易做特征级梯度隔离。FIP v0 仅替换标准 FFN 为 FIP-FFN，其余 Transformer 结构（RMSNorm、RoPE、attention、嵌入）保持不变。

## 2. 普通 FFN（参考）

```
h -> SwiGLU(W_gate, W_up) -> 稠密中间激活 -> W_down -> h'
```

## 3. FIP-FFN 数据流

```
h
 |
 v
SwiGLU 特征槽 (W_gate, W_up)          # [B, T, D] -> [B, T, F]
 |
 v
Top-K 稀疏激活                          # 每个位置仅 K 个特征非零
 |
 v
特征级梯度掩码 (plasticity mask)        # stable=0, shared=eps, plastic/free=1
 |
 v
W_down 解码                             # [B, T, F] -> [B, T, D]
 |
 v
h'
```

## 4. 特征槽定义

一个特征槽 `j` 对应：

| 张量 | 形状 | 说明 |
|---|---|---|
| `W_gate[j, :]` | `[D]` | 门控投影的第 j 行 |
| `W_up[j, :]` | `[D]` | 上投影的第 j 行 |
| `W_down[:, j]` | `[D]` | 下投影的第 j 列 |
| `state[j]` | 标量 | stable/shared/plastic/free |
| `importance[j]` | 标量 | EMA(|activation * gradient|) |
| `version[j]` | 标量 | 该特征被修改的版本号 |

冻结一个特征 = 同时冻结上述三部分权重，禁止更新。

## 5. 特征状态

| 状态 | 行为 |
|---|---|
| stable | 已承载重要旧知识，梯度强制为 0 |
| shared | 多任务共有能力，允许小幅更新（梯度缩放系数 lr_shared） |
| plastic | 当前任务可修改，梯度正常 |
| free | 预留容量，可分配给新知识，梯度正常 |

初版预留 20% 特征为 `free`，从一开始计入模型总参数，训练中途不额外扩容。

## 6. 特征重要性

一阶指标，简单可复现：

```
importance_j = EMA(|activation_j * grad_j|)
```

每个任务结束后：

- 高重要性特征转为 `stable`
- 被多个任务反复使用的转为 `shared`
- 新任务主要写入 `plastic` / `free`
- 保存本轮特征增量，用于回滚测试

## 7. 任务边界约定

- 训练过程允许知道"任务阶段结束"（显式调用 `end_task()`）
- 推理时不提供任务 ID
- 后续再研究无边界自动变化检测

## 8. 回滚

每个任务结束时，记录被修改特征槽的旧权重（W_gate/W_up/W_down 对应切片）与状态。回滚指定任务 = 还原其写入的特征切片（若有重叠，按版本顺序逆序还原）。

## 9. 配置参数（Phase 0）

| 参数 | 值 |
|---|---|
| n_layer | 6 |
| d_model | 256 |
| d_ff | 1024 |
| top_k | 64 |
| free_ratio | 0.2 |
| n_head | 4 |
| context | 256 |
| vocab | 8192（合成任务） |
| importance_ema | 0.99 |
| lr_shared | 0.1（相对缩放） |
