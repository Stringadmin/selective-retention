# Changelog

## 0.3.0 — 2026-09-17

这次发布的重点：**slot 覆盖从"删除旧值"改成"版本化归档"**——抽取失误不再销毁数据，最多是把旧值遮住；历史通过新工具 `recall_history` 对模型可见；所有读路径（召回、注入、跨项目匹配、list/stats）统一走当前投影，被取代的值再也到不了模型上下文。测试 21→27，且能在 Windows 与脱离 DSH checkout 的环境里跑。

- Session-start injection now lists memories oldest first, so the injected list
  reads as a timeline whose newest statement is last - the assembly the reader
  experiments scored best. Injected memories are all current values today, so
  this is convention-setting rather than a fix.
- `remember_fact` returns a `hint` when a fact is stored without an attribute
  key, suggesting the `我的{属性}是{值}` restatement: measured key yield on
  natural phrasing is ~15%, and the template shape is the one reliable path.
- Add a `recall_history` tool: given a natural-language question
  ("我之前的住址是什么"), it routes to the attribute's slot and returns the
  archived version timeline (text, storedAt, archivedAt, current flag), oldest
  first. It probes the project store and the shared user store, project first.
  This closes the last gap of the versioned-supersede port: history was kept on
  disk but invisible to the model.
- Strip "之前的" / "上一次的" / "上次的" as temporal prefixes in `extractSlot`,
  mirroring `igm/gate.py`, so history questions route to the same slot as the
  stored fact.
- **Breaking (storage semantics)**: slot supersede is versioned. A same-slot
  write now closes the previous record's `validTo` and appends a new event with
  `supersedes`, instead of dropping the old value. Nothing is destroyed by an
  extraction mistake any more; `history(slot)` returns the version timeline.
- Fix a real defect found while porting this: `add()` looked up dedup targets by
  **slot** first, so every same-attribute update was rewritten in place and the
  `items.filter(it => it.slot !== slot)` supersede line could never be reached.
  Dedup now matches identical text only; different wording for the same slot is
  a new version.
- All read paths now go through `current()`: `query`, `recall_fact`,
  `igm.memory.list` / `stats`, cross-project experience matching, its topic set,
  and session-start injection. Superseded values can no longer reach the model.
- `size` counts the current projection (unchanged meaning); new `eventCount`
  counts the archive, and `igm.memory.stats` reports both as `stored` / `events`.
  `touch()` no longer refreshes archived records.
- Old store files upgrade on load: records without `validTo` are current, and
  records without `eventId` get sequential ids so `supersedes` stays
  unambiguous. Covered by a new test.
- The cross-language parity comparison (`write_pairs`) now compares the current
  projection; `../reports/slot-ood-baseline.json` was regenerated accordingly.
- Fix `npm test` on Windows: `test/test_igm_plugin.mjs` imported `lib/index.js`
  via a raw absolute path, which the ESM loader rejects
  (`ERR_UNSUPPORTED_ESM_URL_SCHEME`). Use `pathToFileURL(...).href`.
- Add `@deepseek-ai/dsh-tools` as a pinned devDependency so the suite runs
  outside a DSH-linked checkout instead of relying on a host `node_modules`
  symlink.
- Add two parity tests that hold `extractSlot` and the `IgmStore` supersede path
  against the Python oracle in `../reports/slot-ood-baseline.json`, plus three
  `xfail` pins for known slot-extraction defects (false extraction from
  interjections, missed update verbs, tense modifiers splitting one attribute
  into two keys). They behave like `xfail(strict=True)`: fixing a defect fails
  the suite until the pin is removed.
- Count registered vs finished tests and fail if any test never ran.
- Fix `recall_fact` so returned user and project memories update and persist
  `reuseCount` and `lastUsedAt`, just like service-layer queries.
- Refresh matching cross-project `experience` items in their source project
  when they are returned.
- Add regression coverage for both retention updates.
