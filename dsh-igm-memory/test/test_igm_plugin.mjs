// Comprehensive tests for dsh-igm-memory (pure JS, no dsh runtime needed).
// Covers: gate, slot supersede, persistence, injection budget, edge cases.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'

const pluginRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(path.join(pluginRoot, 'lib/index.js'))
const schemastery = require('schemastery')
const mod = await import(path.join(pluginRoot, 'lib/index.js'))
const { IgmStore, extractSlot, importanceScore } = mod

let pass = 0
const test = (name, fn) => {
  try { fn(); pass++; console.log(`  ok  ${name}`) }
  catch (e) { console.error(`FAIL ${name}: ${e.message}`); process.exitCode = 1 }
}

const tmpFile = () => path.join(os.tmpdir(), `igm-test-${Date.now()}-${Math.random().toString(36).slice(2)}.json`)

// ---- slot extraction edges ----
test('slot: fact', () => assert.equal(extractSlot('我的住址是北京。'), '住址'))
test('slot: update', () => assert.equal(extractSlot('更新一下，我的住址现在是深圳了。'), '住址'))
test('slot: query', () => assert.equal(extractSlot('我现在的住址是什么？'), '住址'))
test('slot: long attr rejected', () => assert.equal(extractSlot('我的很长很长的属性名称是值。', 6), null))
test('slot: no slot for filler', () => assert.equal(extractSlot('今天天气不错。'), null))

// ---- gate ----
test('gate: fact high', () => assert.ok(importanceScore('我的职业是软件工程师。') >= 0.6))
test('gate: question zero', () => assert.equal(importanceScore('我的爱好是什么？'), 0))
test('gate: filler low', () => assert.ok(importanceScore('今天天气不错。') < 0.6))

// ---- store: supersede ----
test('supersede: old removed', () => {
  const s = new IgmStore()
  s.add('我的住址是北京。')
  s.add('我的住址现在是深圳了。')
  assert.equal(s.size, 1)
  assert.ok(s.items[0].text.includes('深圳'))
})

// ---- store: length guard ----
test('guard: too-long fact rejected', () => {
  const s = new IgmStore()
  const r = s.add('我的住址是' + '很长很长'.repeat(100) + '。', 0.6, 6, 200)
  assert.equal(r.kept, false)
  assert.equal(r.reason, 'too_long')
})

test('guard: non-string rejected', () => {
  const s = new IgmStore()
  const r = s.add(null)
  assert.equal(r.kept, false)
  assert.equal(r.reason, 'invalid')
})

// ---- consolidation ----
test('consolidate: old never-used low-value dropped, reused kept', () => {
  const s = new IgmStore()
  const now = Date.now()
  // old, never reused, low value -> candidate for removal
  s.items.push({ text: '旧的低价值记忆', slot: null, score: 0.3, ts: now - 60 * 24 * 3600 * 1000, reuseCount: 0 })
  // old but reused -> kept
  s.items.push({ text: '旧但被用过的记忆', slot: null, score: 0.3, ts: now - 60 * 24 * 3600 * 1000, reuseCount: 3 })
  // recent -> kept regardless
  s.items.push({ text: '新的记忆', slot: null, score: 0.3, ts: now - 1000, reuseCount: 0 })
  const removed = s.consolidate(30, 0.5)
  assert.equal(removed, 1)
  assert.equal(s.size, 2)
  assert.ok(s.items.some((i) => i.text.includes('被用过')))
  assert.ok(s.items.some((i) => i.text.includes('新的')))
})

// ---- persistence ----
test('persist: survives reload', () => {
  const f = tmpFile()
  const s1 = new IgmStore(f)
  s1.add('我的住址是北京。')
  s1.add('我的住址现在是深圳了。')
  const s2 = new IgmStore(f) // simulate restart
  assert.equal(s2.size, 1)
  assert.ok(s2.items[0].text.includes('深圳'))
  fs.unlinkSync(f)
})

test('persist: corrupt file tolerated', () => {
  const f = tmpFile()
  fs.writeFileSync(f, '{not valid json')
  const s = new IgmStore(f) // should not throw
  assert.equal(s.size, 0)
  fs.unlinkSync(f)
})

// ---- plugin apply with tools + injection ----
test('plugin: tools + injection armed', async () => {
  const cfg = mod.Config({})
  const tools = []
  const events = {}
  const ctx = {
    provide() {},
    tools: { register(d) { tools.push(d) } },
    on(evt, fn) { events[evt] = fn },
    effect() {},
  }
  // point store at a temp file via env to avoid touching real ~/.dsh
  const origHome = process.env.DSH_HOME
  process.env.DSH_HOME = os.tmpdir()
  mod.apply(ctx, cfg)
  process.env.DSH_HOME = origHome

  const names = tools.map((t) => t.name).sort()
  assert.deepEqual(names, ['recall_fact', 'remember_fact'])
  assert.ok(events['system-prompt/assemble'])

  const rw = tools.find((t) => t.name === 'remember_fact')
  const out1 = await rw.execute({ fact: '我的住址是北京。' })
  assert.equal(out1.stored, true)
  const out2 = await rw.execute({ fact: '我的住址现在是深圳了。' })
  assert.equal(out2.memory.length, 1)
  assert.ok(out2.memory[0].text.includes('深圳'))

  const rd = tools.find((t) => t.name === 'recall_fact')
  const rec = await rd.execute({})
  assert.equal(rec.memory.length, 1)
  assert.ok(rec.memory[0].text.includes('深圳'))

  // injection budget: section carries the current fact
  const assemble = events['system-prompt/assemble']
  const assembled = await assemble({}, {}, async () => ({ sections: [] }))
  const sec = assembled.sections.find((s) => s.name === 'igm-memory')
  assert.ok(sec, 'injection section present')
  assert.ok(sec.text.includes('深圳'))
})

console.log(`\n${pass} tests passed` + (process.exitCode ? ' (with failures)' : ''))
process.exit(process.exitCode || 0)
