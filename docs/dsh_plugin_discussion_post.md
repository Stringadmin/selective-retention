# DSH｜dsh-igm-memory｜给 Agent 记忆装上写入层：重要性闸门 + 版本化覆盖，当前值不矛盾、历史可回溯

## 项目地址 / Project URL

**https://github.com/Stringadmin/dsh-igm-memory**

## 项目介绍 / Introduction

Agent 跨会话记忆有个老大难：**记住了旧值，更新后旧值还留着，矛盾就产生了**。你在会话 A 说"住址是北京"，会话 B 改成"深圳"，很多记忆方案会同时记住两条——之后 agent 回答住址时完全看运气。

`dsh-igm-memory` 给 DSH 的记忆写入路径加了一层 **IGM 写入层**：

1. **重要性闸门**——不是什么都记。问句、闲聊被自动拒绝（问句直接 0 分）。
2. **版本化同属性覆盖（slot supersede）**——"我的住址是北京"之后来了"我的住址现在是深圳"，**北京那条被关闭有效期、留在归档**，记忆里当前只有深圳。矛盾不产生，历史也没被销毁。

配合 DSH 提供：

- **`remember_fact` 工具**——agent 想记事实时走闸门 + 覆盖；抽不到属性键时会返回改写提示
- **`recall_fact` 工具**——读当前记忆（只含当前值）
- **`recall_history` 工具**——问"之前/上一次 X 是什么"时返回该属性的版本时间线（含存储与失效时间）
- **会话启动注入**——新会话开局就能看到此前记住的事实（真正的跨会话记忆），按时间升序呈现
- **`fact / decision / experience` 显式分类**——只有经验类能按主题跨项目复用
- **JSON 持久化**——重启不丢（用户域 `~/.dsh/storages/igm-user.json`，项目域 `igm-project-*.json` 按 session cwd 隔离）
- **遗忘（consolidate）**——长期没用过的旧记忆自动消退

### 端到端验证 / End-to-end verified

```
会话 1: 记住我的住址是北京 → remember_fact 存入 slot=住址
会话 1: 改记住是深圳      → 北京被关闭有效期进归档，当前只剩深圳
会话 2 (新开): 我住址是哪  → 注入 + recall_fact → 答"你的住址现在是深圳"
会话 2: 那我之前住哪？     → recall_history → 答北京（历史可查，未被销毁）
```

## 0.3.0 的变化

- **覆盖从"删除旧值"改成"版本化归档"**：抽取失误不再销毁数据，最多是把旧值遮住。旧存储文件载入时自动升级，不需要清库。
- **新增 `recall_history`**：历史从"只在磁盘上"变成模型可查。
- **所有读路径只走当前投影**：召回、注入、跨项目经验匹配都看不到被取代的值。
- 修了一个真实缺陷：同属性的更新原先会被文本去重逻辑就地改写，导致覆盖分支根本执行不到。
- 27 个回归测试；在 Windows 与脱离 DSH checkout 的环境下也能跑。
- 语义与 Python 库 `igm` 共用同一份实现与跨语言 parity oracle，两边不会漂移。

## 安装 / Install

```sh
dsh plugin --profile web add https://github.com/Stringadmin/dsh-igm-memory
dsh web    # 重启加载
```

## 与 DSH 的集成方式 / How it integrates with DSH

- **Host 插件**（cordis bundle）：`package.json` 声明 `dsh.bundle.patch`，`cordis.patch.yml` 注入插件行
- **`ctx.tools.register(defineTool(...))`** 注册 `remember_fact` / `recall_fact` / `recall_history` 三个模型可见工具
- **`system-prompt/assemble`** 事件：会话启动时把持久化事实按时间升序注入上下文
- **`ctx.provide`** 暴露 `igm.memory.write / query / stats / list / consolidate` 服务，其他插件可调用


## License

MIT
