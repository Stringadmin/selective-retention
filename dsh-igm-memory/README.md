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
| `igm.memory.query` | `(text) -> items` | slot-aware retrieval (top-3), tracks reuse |
| `igm.memory.stats` | `() -> {stored, items}` | inspect current memory |
| `igm.memory.consolidate` | `(maxAgeDays?, minScore?) -> {removed, remaining}` | forget old, never-reused, low-value memories |

## Tests

```sh
# from this directory, with dsh installed (for @deepseek-ai/dsh-tools resolution)
node test/test_igm_plugin.mjs
```

15 cases: slot extraction edges, gate, supersede, length guards, persistence,
corrupt-file tolerance, injection budget, consolidation.

## Config

Set in `cordis.patch.yml` (or a profile patch layer):

```yaml
- id: dsh-igm-memory
  name: dsh-igm-memory
  config:
    enabled: true
    writeThreshold: 0.6
    slotMaxLen: 6
    storeFile: /absolute/path/to/memory.json   # optional per-profile isolation
```

## Multi-profile isolation

By default all profiles share `$DSH_HOME/storages/igm-memory.json`. To keep
memories separate per profile (e.g. web vs headless), set a distinct
`storeFile` in each profile's patch layer — the same plugin instance then
reads/writes only its own memory file.

## End-to-end verification

In a dsh session:

```
> 记住我的住址是北京
> 改记住是深圳
> (start a NEW session)
> 我住址是哪
你的住址现在是深圳
```

The new session sees the persisted, superseded value (Shenzhen only) via
session-start injection — the "knowledge update" case from the research
write-up, working in a real agent harness.

## License

MIT
