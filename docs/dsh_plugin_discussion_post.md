# DSH｜dsh-igm-memory｜给 Agent 记忆装上写入层：重要性闸门 + 同属性覆盖，让跨会话记忆不再矛盾

## 项目地址 / Project URL

**https://github.com/Stringadmin/dsh-igm-memory**

## 项目介绍 / Introduction

Agent 跨会话记忆有个老大难：**记住了旧值，更新后旧值还留着，矛盾就产生了**。你在会话 A 说"住址是北京"，会话 B 改成"深圳"，很多记忆方案会同时记住两条——之后 agent 回答住址时完全看运气。

`dsh-igm-memory` 给 DSH 的记忆写入路径加了一层 **IGM 写入层**：

1. **重要性闸门**——不是什么都记。问句、闲聊被自动拒绝（问句直接 0 分）。
2. **同属性覆盖（slot supersede）**——"我的住址是北京"之后来了"我的住址现在是深圳"，**旧值北京被直接取代**，记忆里只有深圳。矛盾根本不产生。

配合 DSH 提供：

- **`remember_fact` 工具**——agent 想记事实时走闸门 + 覆盖
- **`recall_fact` 工具**——随时读当前记忆
- **会话启动注入**——新会话开局就能看到此前记住的事实（真正的跨会话记忆）
- **JSON 持久化**——重启不丢（`~/.dsh/storages/igm-memory.json`）
- **遗忘（consolidate）**——长期没用过的旧记忆自动消退

### 端到端验证 / End-to-end verified

```
会话 1: 记住我的住址是北京 → remember_fact 存入 slot=住址
会话 1: 改记住是深圳 → 覆盖旧值，记忆只剩深圳
会话 2 (新开): 我住址是哪 → 注入 + recall_fact → 回答"你的住址现在是深圳"
```

新会话直接答对深圳——覆盖 + 跨会话注入在真实 DSH 里完整工作。

## 安装 / Install

```sh
dsh plugin --profile web add https://github.com/Stringadmin/dsh-igm-memory
dsh web    # 重启加载
```

## 与 DSH 的集成方式 / How it integrates with DSH

- **Host 插件**（cordis bundle）：`package.json` 声明 `dsh.bundle.patch`，`cordis.patch.yml` 注入插件行
- **`ctx.tools.register(defineTool(...))`** 注册 `remember_fact` / `recall_fact` 两个模型可见工具
- **`system-prompt/assemble`** 事件：会话启动时把持久化事实注入上下文
- **`ctx.provide`** 暴露 `igm.memory.write / query / stats / consolidate` 服务，其他插件可调用


## License

MIT
