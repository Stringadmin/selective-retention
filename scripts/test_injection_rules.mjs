// Verify: rules injected even with an empty store (fresh profile learns them).
const mod = await import('/mnt/f/FIP-Transformer/dsh-igm-memory/lib/index.js')

const cfg = mod.Config({})
const events = {}
const ctx = {
  provide() {},
  tools: { register() {} },
  on(e, fn) { events[e] = fn },
  effect() {},
}
const origHome = process.env.DSH_HOME
process.env.DSH_HOME = '/tmp/igm-empty-test'
mod.apply(ctx, cfg)
process.env.DSH_HOME = origHome

const assembled = await events['system-prompt/assemble']({}, {}, async () => ({ sections: [] }))
const sec = assembled.sections.find((s) => s.name === 'igm-memory')
console.log('injection present with empty store:', !!sec)
if (sec) {
  console.log('has proactive rules:', sec.text.includes('do not wait for the user'))
  console.log('mentions remember_fact:', sec.text.includes('remember_fact'))
  console.log('--- injected text ---')
  console.log(sec.text.slice(0, 600))
}
