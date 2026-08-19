# dsh-igm-memory

IGM write layer for DeepSeek Harness — an importance-gated memory write gate with slot supersede.

> Community plugin for DeepSeek Harness (DSH). Not an official DeepSeek product.

## What it does

DSH agents accumulate durable memory/instructions over sessions. Without a write
gate, memories pile up: questions get stored as facts, stale values survive
updates, contradictions accumulate. `dsh-igm-memory` sits on the write path:

1. **Gate** — only memories scoring above a threshold are written. Questions
   and chit-chat are rejected (score 0 for interrogatives).
2. **Slot supersede** — a fact about the same attribute ("我的住址是北京" →
   "我的住址现在是深圳") replaces the old value instead of appending.

This is the DSH plugin port of the IGM mechanism from
[selective-retention](https://github.com/Stringadmin/selective-retention)
(see `docs/ARTICLE_DRAFT.md` for the research write-up and evidence).

## Install

```sh
# from this directory (develop against the web profile)
dsh plugin --profile web add ./dsh-igm-memory
dsh --profile web --dump-config   # verify the row loaded
dsh web                           # restart to activate
```

## Exposed services

| Service | Signature | Purpose |
|---|---|---|
| `igm.memory.write` | `(text) -> {kept, item?, score}` | route a candidate memory through the gate |
| `igm.memory.query` | `(text) -> items` | slot-aware retrieval (top-3) |
| `igm.memory.stats` | `() -> {stored, items}` | inspect current memory |

## Config

Set in `cordis.patch.yml` (or a profile patch layer):

```yaml
- id: dsh-igm-memory
  name: dsh-igm-memory
  config:
    enabled: true
    writeThreshold: 0.6
    slotMaxLen: 6
```

## License

MIT
