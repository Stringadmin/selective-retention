// Verify cross-project experience recall:
// project A learns an electron gotcha; project B (also doing electron)
// should surface that experience as a portable lesson.
const mod = await import('/mnt/f/FIP-Transformer/dsh-igm-memory/lib/index.js')

const cfg = mod.Config({})
const events = {}
const tools = []
const ctx = {
  provide() {},
  tools: { register(d) { tools.push(d) } },
  on(e, fn) { events[e] = fn },
  effect() {},
}
const origHome = process.env.DSH_HOME
process.env.DSH_HOME = '/tmp/igm-xp-test'
mod.apply(ctx, cfg)
process.env.DSH_HOME = origHome

const remember = tools.find((t) => t.name === 'remember_fact')
const recall = tools.find((t) => t.name === 'recall_fact')
const assemble = events['system-prompt/assemble']

const PROJECT_A = '/mnt/f/project-a'
const PROJECT_B = '/mnt/f/project-b'
const PROJECT_C = '/mnt/f/project-c'

// A: learn an electron gotcha.
await assemble({}, { agent: { session: { cwd: PROJECT_A } } }, async () => ({ sections: [] }))
await remember.execute({ fact: '这个项目踩过一个坑：electron 打包时 icon 路径要写绝对路径。' })

// B: also electron-related work.
await assemble({}, { agent: { session: { cwd: PROJECT_B } } }, async () => ({ sections: [] }))
await remember.execute({ fact: '这个项目用 electron 做桌面端。' })

// C: unrelated (backend).
await assemble({}, { agent: { session: { cwd: PROJECT_C } } }, async () => ({ sections: [] }))
await remember.execute({ fact: '这个项目后端用 FastAPI。' })

// Back in B: recall should include A's electron experience.
await assemble({}, { agent: { session: { cwd: PROJECT_B } } }, async () => ({ sections: [] }))
const recB = await recall.execute({})
console.log('B memory:', recB.memory.map((m) => m.text).join(' | '))
console.log('B experiences:', JSON.stringify(recB.experiences))

// Injection in B should carry A's gotcha.
const outB = await assemble({}, { agent: { session: { cwd: PROJECT_B } } }, async () => ({ sections: [] }))
const secB = outB.sections.find((s) => s.name === 'igm-memory')
console.log('B injection has A gotcha:', secB.text.includes('icon 路径要写绝对路径'))

// C (backend) should NOT see electron experiences.
await assemble({}, { agent: { session: { cwd: PROJECT_C } } }, async () => ({ sections: [] }))
const outC = await assemble({}, { agent: { session: { cwd: PROJECT_C } } }, async () => ({ sections: [] }))
const secC = outC.sections.find((s) => s.name === 'igm-memory')
console.log('C injection has A gotcha:', secC.text.includes('icon 路径要写绝对路径'))

let fail = 0
const ok = (cond, msg) => { console.log(`  ${cond ? 'ok' : 'FAIL'}  ${msg}`); if (!cond) fail++ }
ok(recB.experiences && recB.experiences.length >= 1 && recB.experiences[0].text.includes('icon'), 'B recalls A electron gotcha')
ok(secB.text.includes('icon 路径要写绝对路径'), 'B injection carries A gotcha')
ok(!secC.text.includes('icon 路径要写绝对路径'), 'C (backend) not polluted')

console.log(fail === 0 ? '\nCROSS-PROJECT EXPERIENCE OK' : `\n${fail} FAILURES`)
process.exit(fail === 0 ? 0 : 1)
