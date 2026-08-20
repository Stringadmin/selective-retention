// Verify high-value memory types (decisions, gotchas) pass the gate,
// and injection carries scope labels + source-citation rule.
import { importanceScore } from '../dsh-igm-memory/lib/index.js'

let fail = 0
const ok = (cond, msg) => { console.log(`  ${cond ? 'ok' : 'FAIL'}  ${msg}`); if (!cond) fail++ }

// Decision/gotcha statements should score above the 0.6 threshold.
const cases = [
  '因为 npm 对 monorepo 支持差，所以选了 pnpm。',
  '这个项目踩过一个坑：electron 打包时 icon 路径要写绝对路径。',
  '当时决定用 Node 20 是因为和 CI 的 runner 版本对齐。',
]
for (const c of cases) {
  const s = importanceScore(c)
  ok(s >= 0.6, `kept (${s.toFixed(2)}): ${c.slice(0, 20)}...`)
}

// A question about why should still score 0.
ok(importanceScore('为什么这个项目用 pnpm？') === 0, 'question still gated')

// Injection: scope labels + source-citation rule present.
const mod = await import('../dsh-igm-memory/lib/index.js')
const events = {}
const tools = []
const ctx = { provide() {}, tools: { register(d) { tools.push(d) } }, on(e, f) { events[e] = f }, effect() {} }
const orig = process.env.DSH_HOME
process.env.DSH_HOME = '/tmp/igm-vis-test'
mod.apply(ctx, mod.Config({}))
process.env.DSH_HOME = orig
// Store a fact first so the injection actually lists it with a scope label.
const remember = tools.find((t) => t.name === 'remember_fact')
await remember.execute({ fact: '这个项目用 pnpm。' })
const out = await events['system-prompt/assemble']({}, { agent: { session: { cwd: '/p/x' } } }, async () => ({ sections: [] }))
const sec = out.sections.find((s) => s.name === 'igm-memory')
ok(sec.text.includes('[project]'), 'project scope label in injection')
ok(sec.text.includes('[user]'), 'user scope label present in legend')
ok(sec.text.includes('根据你之前说的') || sec.text.includes('source'), 'source-citation rule present')

console.log(fail === 0 ? '\nHIGH-VALUE MEMORY + VISIBILITY OK' : `\n${fail} FAILURES`)
process.exit(fail === 0 ? 0 : 1)
