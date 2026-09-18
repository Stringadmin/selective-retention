# Changelog

## 0.5.0 — 2026-09-18

这一版把两条早已分叉的实现合成一条线。本仓库的 `0.4.0-experimental`（标签
`v0.4.0-experimental`，2026-08-24）加了**写入信任边界**：可疑的控制指令被隔离待审、
凭据类写入被拒绝、记录带上 provenance / trust / visibility。另一条线在研究主仓
`selective-retention` 的 `dsh-igm-memory/` 目录里演进（从未发布到本仓库，那边自称
0.3.0），把 **slot 覆盖改成了默认的版本化归档**，并补上 `recall_history`。0.5.0 同时
保留两者，并解决三处真正的语义冲突（见"有意的取舍"）。回归测试 20 → 36。

### 归档成为默认语义（承自研究主仓那条线）

- **Breaking (storage semantics)**: slot supersede is versioned by default. A
  same-slot write closes the previous record's `validTo` and appends a new event
  with `supersedes`, instead of dropping the old value. Nothing is destroyed by
  an extraction mistake any more.
- Fix a real defect found while porting this: `add()` looked up dedup targets by
  **slot** first, so every same-attribute update was rewritten in place and the
  supersede branch could never be reached. Dedup now matches identical text only;
  different wording for the same slot is a new version.
- All read paths now go through `current()`, which is both the current projection
  *and* the accepted set: `query`, `recall_fact`, `recall_history`,
  `igm.memory.list` / `stats` / `history`, cross-project experience matching, its
  topic set, and session-start injection. Neither a superseded value nor a record
  the trust boundary has not accepted can reach the model.
- New model-facing `recall_history` tool: given a natural-language question
  ("我之前的住址是什么"), it routes to the attribute's slot and returns the version
  timeline, oldest first. `recall_fact` keeps the `mode` / `query` / `limit`
  parameters from 0.3 and reads the same timeline.
- `size` counts the current projection; `eventCount` counts the archive, and
  `igm.memory.stats` reports both as `stored` / `events` plus `pendingReview` and
  `rejectedWrites`. `touch()` no longer refreshes archived records.
- Session-start injection lists memories oldest first, so the injected list reads
  as a timeline whose newest statement is last — the assembly the reader
  experiments scored best.
- `remember_fact` returns a `hint` when a fact is stored without an attribute key:
  measured key yield on natural phrasing is ~15%, and the `我的{属性}是{值}`
  template is the one reliable path.
- A 0.3/0.4 store file upgrades on first load: its `events` archive is read (not
  the `items` current-snapshot), and `mode` is still written in the file header.
  Records with no `validTo` are current, and records with no `eventId` get
  sequential ids so `supersedes` stays unambiguous. Covered by tests.
- Strip "之前的" / "上一次的" / "上次的" as temporal prefixes in `extractSlot`,
  mirroring `igm/gate.py`, so history questions route to the same slot as the
  stored fact.
- Add parity tests that hold `extractSlot` and the `IgmStore` supersede path
  against the Python oracle, plus `xfail` pins for known slot-extraction defects
  (false extraction from interjections, missed update verbs, tense modifiers
  splitting one attribute into two keys). They behave like `xfail(strict=True)`:
  fixing a defect fails the suite until the pin is removed. The oracle now ships
  as `test/fixtures/slot-ood-baseline.json` so the standalone repository can run
  these tests; the mono-repo CI compares the two copies.
- Count registered vs finished tests and fail if any test never ran.
- Fix `npm test` on Windows (`ERR_UNSUPPORTED_ESM_URL_SCHEME` from a raw absolute
  path; use `pathToFileURL(...).href`) and pin `@deepseek-ai/dsh-tools`
  (0.1.0-rc.6) as a devDependency so `npm ci && npm test` works from a clean
  checkout outside a DSH-linked checkout.

### 有意的取舍（与 `0.4.0-experimental` 不同）

1. `versioned: true` no longer selects a second store class: archiving is the
   default and `VersionedIgmStore` is gone. Set `supersede: "delete"` — or keep
   the old `versioned: false` key, which maps to the same thing — to get the
   destructive current-only store back.
2. 0.3/0.4 disabled `consolidate()` in versioned mode (it returned 0). This
   release keeps the explicit forgetting service advertised in the README: nothing
   decays on its own, and a value that was recalled again survives. A host that
   must never trim the archive simply does not call it.
3. The slot-prefix table is held identical to `igm/gate.py` again. 0.4 had added
   bare forms (`上次`, `上一次`, `之前`, `历史`), which forked the JS port away from
   the Python oracle it is tested against; `之前的X` / `上一次的X` still route,
   `我上次用的X` no longer does. A test pins the parity either way.

## 0.4.0-experimental — 2026-08-24

- Add a guarded-write trust boundary. High-signal prompt-control patterns are
  quarantined for explicit host review and never reach normal retrieval,
  prompt injection, or cross-project sharing by default.
- Reject credential-shaped writes without persisting their text; retain only a
  bounded fingerprint audit record.
- Add structured provenance, trust state, and visibility metadata to memory
  records. New project experiences are project-only unless a service caller
  explicitly sets `visibility: "cross-project"`.
- Add host services `igm.memory.audit` (redacted by default) and
  `igm.memory.review`. Deliberately do not expose approval as a model-facing
  tool.
- Persist pending/rejected review records alongside current state or versioned
  history, while preserving the prior accepted slot value until approval.
- Expand the regression suite with guarded-write, credential, review,
  visibility, and cross-project isolation scenarios.

## 0.3.0 — 2026-08-20

- Add opt-in `versioned: true` storage: same-slot writes now preserve an
  append-only event archive while the current-state view remains compact.
- Add `recall_fact` modes `previous` and `history`, plus the
  `igm.memory.history` service, for explicit historical queries.
- Preserve current-only injection in versioned mode so archived values cannot
  silently compete with active state.
- Migrate legacy v0.2 JSON current items into versioned events on first
  versioned mutation, while retaining a current-value compatibility snapshot.
- Fix the Windows ESM test loader by converting the absolute plugin path with
  `pathToFileURL()`.
- Add a lockfile and DSH test dependency so `npm ci && npm test` works from a
  clean checkout; expand regression coverage from 16 to 20 tests.
- Fix `recall_fact` so returned user and project memories update and persist
  `reuseCount` and `lastUsedAt`, just like service-layer queries.
- Refresh matching cross-project `experience` items in their source project
  when they are returned.
- Add regression coverage for both retention updates.
