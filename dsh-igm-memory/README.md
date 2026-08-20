# dsh-igm-memory

Typed, importance-gated memory tools for DeepSeek Harness (DSH), with slot
supersede and per-project isolation.

> Community plugin for DSH. Not an official DeepSeek product.

## Behavior

The plugin registers `remember_fact` and `recall_fact` and injects rules that
tell the agent when to use them.

1. An importance gate rejects questions, chit-chat, and oversized candidates.
2. Slot supersede replaces an older value for the same attribute.
3. User memories are shared; project memories are isolated by session cwd.
4. Every item is typed as `fact`, `decision`, or `experience`.
5. Only project-scoped `experience` items can transfer to another project, and
   only when the projects share a detected topic.
6. Project discovery is persisted, so cross-project experience recall survives
   a DSH restart.
7. Both `recall_fact` and `igm.memory.query` record `reuseCount` and
   `lastUsedAt`. A `recall_fact` response already includes the count from that
   recall. Consolidation uses recent write/recall activity, so stale accepted
   memories can actually expire.

Old JSON files remain readable. Missing metadata is derived when loaded and is
written back on the next mutation.

## Install

From the plugin directory:

```sh
dsh plugin --profile web add .
dsh --profile web --dump-config
dsh web
```

Restart the active DSH process after changing the plugin.

## Config

```yaml
- id: dsh-igm-memory
  name: dsh-igm-memory
  config:
    enabled: true
    writeThreshold: 0.6
    slotMaxLen: 6
    maxFactLen: 200
    maxInjectionBytes: 2048
    storeFile: /absolute/path/to/profile-user-memory.json
```

`storeFile` changes the user-memory file only. Project stores remain under
`$DSH_HOME/storages` so they can be discovered across sessions.

## Storage

Default files under `$DSH_HOME/storages`:

| File | Contents |
|---|---|
| `igm-user.json` | User-scoped memories shared across projects |
| `igm-project-<hash>.json` | Memories isolated to one project cwd |
| `igm-projects.json` | Project hash-to-cwd discovery index |
| `igm-project-unknown.json` | Project memories written without a cwd |

Each item includes its text, slot, score, timestamp, topics, type, scope,
project identity, provenance, reuse count, and last-used timestamp.

## Services

| Service | Signature |
|---|---|
| `igm.memory.write` | `(text, { cwd?, type?, provenance? })` |
| `igm.memory.query` | `(text, cwd?)` |
| `igm.memory.stats` | `(cwd?)` |
| `igm.memory.list` | `(cwd?)` |
| `igm.memory.consolidate` | `(maxAgeDays = 30, minScore = 1, cwd?)` |

Service callers should always pass `cwd` for project-scoped operations. A
project write without cwd is intentionally kept in the unknown-project store,
never in the shared user store.

## Tests

Run from a DSH-linked checkout where the host provides
`@deepseek-ai/dsh-tools`:

```sh
npm test
```

The 16-test suite uses unique temporary directories and never reads or writes
the real `~/.dsh/storages`. It covers gating, supersede, persistence, corrupt
files, typed migration, concurrent session routing, restart discovery,
project-fact isolation, reuse persistence through both query and
`recall_fact`, practical forgetting, `storeFile`, and prompt injection.

## Boundaries and privacy

This plugin is a guarded tool and service layer. It does not intercept a DSH
memory path or another plugin that writes durable data without calling
`remember_fact` or `igm.memory.write`.

Memory and the project cwd registry are stored as plaintext JSON. Do not store
secrets. Apply normal filesystem permissions and backup policy, and stop DSH
before manually deleting or editing these files.

The implementation demonstrates selective retention and controlled
cross-project lesson reuse. It is not evidence that every standard RAG system
has the same failure rate or that the approach is universally superior.

## License

MIT
