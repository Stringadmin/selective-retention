// Simulate cordis loading the plugin: verify name/Config/apply exports work
// together without the full dsh runtime. This catches the exact class of
// error the web profile hit ("cannot get property config without inject").
import { createRequire } from 'node:module'
const require = createRequire('/mnt/f/FIP-Transformer/dsh-igm-memory/lib/index.js')
const schemastery = require('schemastery')

const mod = await import('/mnt/f/FIP-Transformer/dsh-igm-memory/lib/index.js')

console.log('exports:', Object.keys(mod))
console.log('name:', mod.name)
console.log('inject:', JSON.stringify(mod.inject))

// Validate the Config schema the way cordis does.
const parsed = mod.Config({ enabled: true, writeThreshold: 0.6, slotMaxLen: 6 })
console.log('Config parse ok:', JSON.stringify(parsed))

// Apply with a minimal fake ctx + config (cordis passes config as 2nd arg).
const services = {}
const fakeCtx = {
  config: parsed,
  provide(name, fn) { services[name] = fn },

  on() {},
  effect() {},
}
mod.apply(fakeCtx, parsed)

// Exercise the services exactly as the agent loop would.
console.log('\n-- write gate --')
console.log('fact 住址:', JSON.stringify(services['igm.memory.write']('我的住址是北京。')))
console.log('update 住址:', JSON.stringify(services['igm.memory.write']('我的住址现在是深圳了。')).slice(0, 120))
console.log('filler:', JSON.stringify(services['igm.memory.write']('今天天气不错。')))
console.log('question:', JSON.stringify(services['igm.memory.write']('我的爱好是什么？')).slice(0, 80))
console.log('\n-- stats --')
console.log(JSON.stringify(services['igm.memory.stats']()))
console.log('\n-- query --')
console.log(JSON.stringify(services['igm.memory.query']('我现在的住址是什么？')).slice(0, 200))

const stored = services['igm.memory.stats']().stored
const hasOld = services['igm.memory.stats']().items.some((i) => i.text.includes('北京'))
console.log('\nverify: stored=%d, old_value_present=%s', stored, hasOld)
if (stored >= 1 && !hasOld) console.log('\nPLUGIN LOAD + SERVICES OK')
else { console.error('FAIL'); process.exit(1) }
