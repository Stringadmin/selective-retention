// Verify per-project memory isolation: two different cwds get separate stores.
const mod = await import('/mnt/f/FIP-Transformer/dsh-igm-memory/lib/index.js')

const cfg = mod.Config({ perProject: true })
const events = {}
const tools = []
const ctx = {
  provide() {},
  tools: { register(d) { tools.push(d) } },
  on(e, fn) { events[e] = fn },
  effect() {},
}
const origHome = process.env.DSH_HOME
process.env.DSH_HOME = '/tmp/igm-perproj-test'
mod.apply(ctx, cfg)
process.env.DSH_HOME = origHome

const remember = tools.find((t) => t.name === 'remember_fact')
const recall = tools.find((t) => t.name === 'recall_fact')
const assemble = events['system-prompt/assemble']

const PROJECT_A = '/mnt/f/project-a'
const PROJECT_B = '/mnt/f/project-b'

// Session A starts, remembers its own fact.
await assemble({}, { agent: { session: { cwd: PROJECT_A } } }, async () => ({ sections: [] }))
const rA = await remember.execute({ fact: '这个项目用 pnpm。' })
console.log('A remember:', JSON.stringify(rA))

// Session B starts (different project), remembers a different fact.
await assemble({}, { agent: { session: { cwd: PROJECT_B } } }, async () => ({ sections: [] }))
await remember.execute({ fact: '这个项目用 npm。' })

// Session A again: must only see its own fact.
await assemble({}, { agent: { session: { cwd: PROJECT_A } } }, async () => ({ sections: [] }))
const recallA = await recall.execute({})
const textA = recallA.memory.map((m) => m.text).join(' | ')
console.log('A 记忆:', textA)
console.log('A 含 pnpm（对）:', textA.includes('pnpm'))
console.log('A 不含单独 npm（对）:', !textA.includes('用 npm') && !textA.includes('npm。'))

// Session B: must only see its own fact.
await assemble({}, { agent: { session: { cwd: PROJECT_B } } }, async () => ({ sections: [] }))
const recallB = await recall.execute({})
const textB = recallB.memory.map((m) => m.text).join(' | ')
console.log('B 记忆:', textB)
console.log('B 含 npm（对）:', textB.includes('npm'))
console.log('B 不含 pnpm（对）:', !textB.includes('pnpm'))

// Injection section also isolated.
const outA = await assemble({}, { agent: { session: { cwd: PROJECT_A } } }, async () => ({ sections: [] }))
const secA = outA.sections.find((s) => s.name === 'igm-memory')
console.log('A 注入含 pnpm:', secA.text.includes('pnpm'))
console.log('A 注入不含 npm 词:', !secA.text.includes('用 npm'))
console.log('--- A 注入文本 ---')
console.log(secA.text)

const ok = textA.includes('pnpm') && !textA.includes('用 npm') && textB.includes('npm') && !textB.includes('pnpm')
console.log(ok ? '\nPER-PROJECT ISOLATION OK' : '\nFAIL')
process.exit(ok ? 0 : 1)
