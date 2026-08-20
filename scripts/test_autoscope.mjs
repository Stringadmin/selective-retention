// Verify auto-scope routing (zero config):
// - user facts ("我的...") go to the shared store, visible in every project
// - project facts ("这个项目...") go to a per-cwd store, isolated per project
const mod = await import('/mnt/f/FIP-Transformer/dsh-igm-memory/lib/index.js')

const cfg = mod.Config({})   // no perProject flag — routing is automatic
const events = {}
const tools = []
const ctx = {
  provide() {},
  tools: { register(d) { tools.push(d) } },
  on(e, fn) { events[e] = fn },
  effect() {},
}
const origHome = process.env.DSH_HOME
process.env.DSH_HOME = '/tmp/igm-auto-test'
mod.apply(ctx, cfg)
process.env.DSH_HOME = origHome

const remember = tools.find((t) => t.name === 'remember_fact')
const recall = tools.find((t) => t.name === 'recall_fact')
const assemble = events['system-prompt/assemble']

const PROJECT_A = '/mnt/f/project-a'
const PROJECT_B = '/mnt/f/project-b'

// Project A: user fact + project fact.
await assemble({}, { agent: { session: { cwd: PROJECT_A } } }, async () => ({ sections: [] }))
await remember.execute({ fact: '我的常用语言是 Python。' })
await remember.execute({ fact: '这个项目用 pnpm。' })

// Project B: only its own project fact; user fact should still be visible.
await assemble({}, { agent: { session: { cwd: PROJECT_B } } }, async () => ({ sections: [] }))
await remember.execute({ fact: '这个项目用 npm。' })

// Recall in A: user + A project.
await assemble({}, { agent: { session: { cwd: PROJECT_A } } }, async () => ({ sections: [] }))
const recallA = await recall.execute({})
const textA = recallA.memory.map((m) => m.text).join(' | ')
console.log('A 记忆:', textA)
console.log('A 有用户事实（Python）:', textA.includes('Python'))
console.log('A 有自己项目事实（pnpm）:', textA.includes('pnpm'))
console.log('A 无 B 项目事实（npm）:', !textA.includes('用 npm'))

// Recall in B: user + B project only.
await assemble({}, { agent: { session: { cwd: PROJECT_B } } }, async () => ({ sections: [] }))
const recallB = await recall.execute({})
const textB = recallB.memory.map((m) => m.text).join(' | ')
console.log('B 记忆:', textB)
console.log('B 有用户事实（Python）:', textB.includes('Python'))
console.log('B 有自己项目事实（npm）:', textB.includes('用 npm'))
console.log('B 无 A 项目事实（pnpm）:', !textB.includes('pnpm'))

// Injection in A includes both scopes.
const outA = await assemble({}, { agent: { session: { cwd: PROJECT_A } } }, async () => ({ sections: [] }))
const secA = outA.sections.find((s) => s.name === 'igm-memory')
console.log('A 注入含 Python:', secA.text.includes('Python'))
console.log('A 注入含 pnpm:', secA.text.includes('pnpm'))

const ok = textA.includes('Python') && textA.includes('pnpm') && !textA.includes('用 npm')
  && textB.includes('Python') && textB.includes('用 npm') && !textB.includes('pnpm')
console.log(ok ? '\nAUTO-SCOPE ROUTING OK (user shared, project isolated)' : '\nFAIL')
process.exit(ok ? 0 : 1)
