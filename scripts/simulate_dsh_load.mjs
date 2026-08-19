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
const tools = []
const fakeCtx = {
  config: parsed,
  provide(name, fn) { services[name] = fn },
  tools: { register(def) { tools.push(def) } },
  on() {},
  effect() {},
}
mod.apply(fakeCtx, parsed)
console.log('tools registered:', tools.map((t) => t.name).join(', '))

// Exercise the tool exactly as the agent loop would (via defineTool execute).
console.log('\n-- remember_fact tool --')
const tool = tools.find((t) => t.name === 'remember_fact')
async function runTool(fact) {
  const out = await tool.execute({ fact })
  return out
}

// Exercise the tool end-to-end: gate + slot supersede via the model-facing API.
console.log('\n-- remember_fact tool --')
const r1 = await runTool('我的住址是北京。')
console.log('fact 住址:', JSON.stringify(r1))
const r2 = await runTool('我的住址现在是深圳了。')
console.log('update 住址:', JSON.stringify(r2))
const r3 = await runTool('今天天气不错。')
console.log('filler:', JSON.stringify(r3))
const r4 = await runTool('我的爱好是什么？')
console.log('question:', JSON.stringify(r4))

const stored = services['igm.memory.stats']().stored
const hasOld = services['igm.memory.stats']().items.some((i) => i.text.includes('北京'))
const slotOk = services['igm.memory.stats']().items.every((i) => i.slot !== '住址' || i.text.includes('深圳'))
console.log('\nverify: stored=%d, old_value_present=%s, current_value_ok=%s', stored, hasOld, slotOk)
if (stored >= 1 && !hasOld && slotOk) console.log('\nPLUGIN LOAD + TOOL OK')
else { console.error('FAIL'); process.exit(1) }

