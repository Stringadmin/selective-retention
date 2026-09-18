# dsh-igm-memory

Typed, importance-gated and guarded memory tools for DeepSeek Harness (DSH),
with versioned slot supersede, per-project isolation, and a host-reviewable
trust boundary.

> Community plugin for DSH. Not an official DeepSeek product.

## Behavior

The plugin registers `remember_fact`, `recall_fact` and `recall_history`, and
injects rules that tell the agent when to use them.

1. An importance gate rejects questions, chit-chat, and oversized candidates.
2. Slot supersede is versioned by default: a new value for the same attribute
   becomes the only current one, and the superseded record keeps its `validTo` in
   the JSON store instead of being deleted. Retrieval, cross-project matching,
   and session-start injection read the current projection only — which is also
   the accepted set, so an archived value and a quarantined write are both
   invisible to the model. `recall_history` exposes the archived timeline for one
   attribute when the model asks what a value used to be. Set
   `supersede: "delete"` for the pre-0.5 destructive store.
3. User memories are shared; project memories are isolated by session cwd.
4. Every item is typed as `fact`, `decision`, or `experience`.
5. Only accepted project-scoped `experience` items explicitly marked
   `visibility: "cross-project"` can transfer to another project, and only
   when the projects share a detected topic.
6. Project discovery is persisted, so cross-project experience recall survives
   a DSH restart.
7. `recall_fact` supports `mode=previous` and `mode=history` with a `query`, and
   `recall_history` returns the same timeline routed from a natural-language
   question; prompt injection always receives only current, accepted values.
8. Both `recall_fact` and `igm.memory.query` record `reuseCount` and
   `lastUsedAt`. A `recall_fact` response already includes the count from that
   recall. `igm.memory.consolidate` is only ever called by the host: it drops
   rows whose last write/recall is older than the age window and whose score is
   below `minScore`, so a value that keeps being recalled survives while an
   unrecalled archived version is the first to go.
9. With `securityEnabled: true` (the default), credential-shaped writes are
   rejected without retaining their text. High-signal prompt-control writes
   are quarantined for host review and cannot be retrieved, injected, or
   shared until explicitly accepted.

Old JSON files remain readable. Missing metadata is derived when loaded and is
written back on the next mutation: records without `validTo` load as current, and
records without `eventId` receive sequential ids so `supersedes` stays unambiguous.

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
    # Superseded slot values are archived, not dropped. Set "delete" — or the
    # equivalent legacy key `versioned: false` — for the pre-0.5 behavior that
    # overwrites the old value in place.
    supersede: archive
    # Guard writes with deterministic credential/control-instruction rules.
    securityEnabled: true
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

Each current item includes its text, slot, score, timestamp, topics, type,
scope, project identity, structured provenance, visibility, trust state,
reuse count, and last-used timestamp. Pending control-instruction candidates
live in a separate `quarantine` collection and do not alter an accepted slot
until a trusted host approves them. Credential rejections retain only a
bounded fingerprint audit entry, never the rejected text.
When `supersede: "archive"` (the default), `items` is the append-only event
archive: every record carries `eventId`, `validFrom`, `validTo`, and
`supersedes`, and the current state is the projection of the records whose
`validTo` is `null`. A `supersede: "delete"` store keeps only current records in
`items` and closes overwriting values in place. A 0.4-experimental file, which
split the data into a current `items` snapshot plus an `events` archive, is read
from `events` and rewritten in the single-archive layout on the next mutation;
v0.2/0.3 JSON is upgraded the same way.

Do not downgrade to a pre-0.5 plugin after writing with this one: older code
reads `items` as the current-state snapshot, so it will treat the whole archive
as current and its next write will not preserve the timeline. Back up the JSON
file before a downgrade.

## Versioned history

Every same-slot update keeps the value it replaces, because the version
timeline is what makes an extraction mistake recoverable: the cost is storage
growth, and the alternative is a store that silently destroys the old value.
Set `supersede: "delete"` if you want the pre-0.5 destructive behavior.

For a current-state question, use the ordinary recall path. For an explicit
historical question, the agent calls:

```text
recall_fact({ query: "我之前的住址是什么？", mode: "previous" })
recall_fact({ query: "我的住址历史是什么？", mode: "history", limit: 10 })
```

`previous` returns the one version before the active value; `history` returns
the slot timeline oldest-first. Both require a query that resolves to a slot.
Historical events are never injected into the system prompt automatically,
so an outdated value cannot silently compete with the current state.

## Services

| Service | Signature |
|---|---|
| `igm.memory.write` | `(text, { cwd?, type?, provenance?, visibility? })` |
| `igm.memory.query` | `(text, cwd?)` |
| `igm.memory.history` | `(queryOrSlot, cwd?, limit = 10)` — empty for a `supersede: "delete"` store |
| `igm.memory.stats` | `(cwd?)` |
| `igm.memory.list` | `(cwd?)` |
| `igm.memory.audit` | `(cwd?, { includeText?: false })` — host-only audit, redacted by default |
| `igm.memory.review` | `(reviewId, "accept" \| "reject", cwd?, { reviewer? })` — host-only approval |
| `igm.memory.consolidate` | `(maxAgeDays = 30, minScore = 1, cwd?)` |

Service callers should always pass `cwd` for project-scoped operations. A
project write without cwd is intentionally kept in the unknown-project store,
never in the shared user store.

`visibility` is deliberately not exposed on the model-facing
`remember_fact` tool. New project memories default to `project`; a trusted
host or integration must explicitly choose `cross-project` for a reusable
experience. User-scope memory is always `private` to the local profile.
For backward compatibility, pre-0.4 project `experience` records without a
visibility field load as legacy cross-project records; review or rewrite them
with `visibility: "project"` if you need the stricter new default everywhere.

## Trust boundary

The guarded write path is a deterministic safety control, not an LLM-based
classifier and not a universal prompt-injection solution:

- Credential-shaped values are rejected before persistence. The audit record
  contains a one-way fingerprint, length, reason, timestamp, and provenance.
- Obvious control-instruction patterns are stored only in `quarantine` with
  `trust: review`. They neither replace a current slot nor enter retrieval,
  cross-project experience transfer, history results, or prompt injection.
- Only a trusted host can call `igm.memory.review`. Approval is intentionally
  not a model-facing tool, so an agent cannot approve its own suspicious
  write. An approved candidate then follows normal slot/version semantics.
- `igm.memory.audit` defaults to fingerprints and metadata; pass
  `{ includeText: true }` only from a trusted review UI or host process.

This implements a narrow, testable rule defense. It does not authenticate
service callers, detect all adversarial wording, sanitize every imported file,
or make plaintext local storage secure against a user/process that can read
the store file. See [the system blueprint](docs/AGENT_MEMORY_STACK.md) and
[the threat model](docs/TRUST_MODEL.md) for the intended boundary.

## Tests

```sh
npm ci               # schemastery + @deepseek-ai/dsh-tools (dev), both public
npm test
```

This also runs on Windows and outside a DSH checkout: `schemastery` and
`@deepseek-ai/dsh-tools` install from npm instead of relying on the host's
symlinked `node_modules`.

The 36-test suite uses unique temporary directories and never reads or writes the
real `~/.dsh/storages`. It covers gating, versioned supersede, dedup versus
update, history routing via `versionTimeline`, `recall_history` and
`recall_fact mode=previous/history`, both supersede modes, oldest-first
injection order, the no-slot rephrase hint, persistence, corrupt files, the
0.3/0.4 store-file upgrade, v1 store upgrade, typed migration, concurrent session
routing, restart discovery, project-fact isolation, reuse persistence through
both query and `recall_fact`, explicit consolidation, `storeFile`, prompt
injection, credential non-persistence, quarantine, host review, and cross-project
visibility. No `xfail` pin is left open — the three `extractSlot` defects they
used to hold open were fixed and are now ordinary regression assertions.

### Slot-rule parity with Python

The slot rule exists twice: `extract_slot` in `igm/gate.py` and `extractSlot` in
`lib/index.js` (`memory_arch` reuses the Python one). Two parity tests hold this
copy against the oracle fixture:

- `extractSlot agrees with igm/gate.py on the labelled OOD corpus` — all 31
  labelled utterances
- `write-layer outcome matches igm/store.py on the labelled pairs` — the 6
  update/collision write pairs, end to end through `IgmStore.add`

In the research mono-repo the live oracle (`reports/slot-ood-baseline.json`,
generated by `python -m memory_arch.run_slot_ood`) is read directly; standalone
checkouts fall back to the vendored copy at
`test/fixtures/slot-ood-baseline.json`, which that repo's CI compares byte for
byte against the generated one.

`write_pairs` compares the **current projection**, which is what every read path
exposes. The archive behind it is asserted separately on each side
(`test_a_superseded_value_is_masked_not_destroyed` in `tests/test_slot_ood.py`,
`slot supersede masks the old value and keeps it archived` here).

## Storage microbenchmark

Run the local storage-only benchmark with:

```sh
npm run benchmark    # IGM_BENCH_FACTS / IGM_BENCH_UPDATES / IGM_BENCH_READS
```

On Windows x64 / Node 24.18.0, with 300 active slot facts, 100 same-slot updates
and 100 reads, the merged 0.5.0 store produced:

| Mode | Active / events | JSON bytes | Write p50 / p95 | Update p50 / p95 | Current-query p50 / p95 | History p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| `supersede: "delete"` | 300 / 300 | 243,308 | 1.303 / 2.219 ms | 2.031 / 2.556 ms | 1.878 / 2.356 ms | 0.009 / 0.019 ms |
| `supersede: "archive"` (default) | 300 / 400 | 324,802 | 1.303 / 2.235 ms | 2.319 / 2.982 ms | 2.435 / 3.290 ms | 0.005 / 0.011 ms |

Archiving costs about 33% more bytes on disk and roughly 0.5 ms p50 on a
current-query over this load; the history lookup itself stays under 0.02 ms
because it scans a per-slot slice of the same array. In `delete` mode the
"history" column is a one-row timeline, which is all that store can still answer.
This is one run of a local JSON-store microbenchmark, not DSH boot latency,
embedding latency, LLM latency, task success, or a RAG benchmark. The source is
[scripts/benchmark_igm_store.mjs](scripts/benchmark_igm_store.mjs).

## Boundaries and privacy

This plugin is a guarded tool and service layer. It does not intercept a DSH
memory path or another plugin that writes durable data without calling
`remember_fact` or `igm.memory.write`.

Memory, quarantined non-secret candidates, and the project cwd registry are
stored as plaintext JSON. Credential-shaped writes are rejected, but this is
not a data-loss-prevention system: do not store secrets, apply normal
filesystem permissions and backup policy, and stop DSH before manually
deleting or editing these files.

Versioned history is an audit feature, not a claim that the plugin performs
online model training or beats RAG. It retains accepted facts exactly as they
were written; it cannot correct a bad extraction, a harmful write, or a wrong
slot automatically.

The implementation demonstrates selective retention, auditable current-state
updates, and controlled cross-project lesson reuse. It is not evidence that
every standard RAG system has the same failure rate, that this approach is
universally superior, or that it autonomously trains model parameters.

## License

MIT
