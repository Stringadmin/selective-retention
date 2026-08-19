// Local sanity check for the IGM JS port (runs standalone with node).
import { importanceScore, extractSlot, IgmStore } from '../dsh-igm-memory/lib/index.js'

let fail = 0
const ok = (cond, msg) => {
  if (cond) console.log(`  ok  ${msg}`)
  else { console.error(`FAIL ${msg}`); fail++ }
}

console.log('=== slot extraction ===')
ok(extractSlot('我的住址是北京。') === '住址', 'fact 住址')
ok(extractSlot('更新一下，我的住址现在是深圳了。') === '住址', 'update 住址')
ok(extractSlot('我现在的住址是什么？') === '住址', 'query 住址')
ok(extractSlot('我的宠物是什么？') === '宠物', 'query 宠物')
ok(extractSlot('今天天气不错。') === null, 'filler no slot')

console.log('=== importance gate ===')
ok(importanceScore('我的职业是软件工程师。') >= 0.6, 'fact high')
ok(importanceScore('今天天气不错。') < 0.6, 'filler low')
ok(importanceScore('我的爱好是什么？') === 0, 'question zero')

console.log('=== store: slot supersede ===')
const store = new IgmStore()
const r1 = store.add('我的住址是北京。')
ok(r1.kept === true, 'first addr kept')
const r2 = store.add('我的住址现在是深圳了。')
ok(r2.kept === true, 'update kept')
ok(store.size === 1, 'superseded -> size 1')
ok(store.items[0].text.includes('深圳'), 'only new value')

console.log('=== store: gate filters ===')
const r3 = store.add('今天天气不错。')
ok(r3.kept === false && r3.reason === 'gate', 'filler filtered')
const r4 = store.add('我的爱好是什么？')
ok(r4.kept === false, 'question filtered')

console.log('=== store: slot-aware query ===')
const q = store.query('我现在的住址是什么？')
ok(q.length >= 1 && q[0].slot === '住址', 'query hits addr slot')

console.log(fail === 0 ? '\nALL PASS' : `\n${fail} FAILURES`)
process.exit(fail === 0 ? 0 : 1)
