// Verify project-fact slot fallback + supersede for the exact failing case.
import { IgmStore } from '../dsh-igm-memory/lib/index.js'

let fail = 0
const ok = (cond, msg) => { console.log(`  ${cond ? 'ok' : 'FAIL'}  ${msg}`); if (!cond) fail++ }

const s = new IgmStore(null)

// The exact case that errored in production: no 我的X是Y shape.
const r = s.add('这个项目使用 pnpm 作为包管理器。')
ok(r.kept === true, 'kept (not gated)')
ok(r.item && r.item.slot === '包管理器', `fallback slot extracted: ${r.item && r.item.slot}`)

// Update same attribute -> supersede.
const r2 = s.add('这个项目使用 bun 作为包管理器。')
ok(r2.kept === true && s.size === 1, 'superseded to size 1')
ok(r2.item.slot === '包管理器' && r2.item.text.includes('bun'), 'new value kept, old dropped')

// Without fallback marker words, slot is null but still stored safely.
const r3 = s.add('后端服务部署在 47.110.225.76。')
ok(r3.kept === true, 'other project fact kept')
ok(r3.item.slot === null || typeof r3.item.slot === 'string', `slot safe (${r3.item.slot})`)

console.log(fail === 0 ? '\nPROJECT SLOT FIX OK' : `\n${fail} FAILURES`)
process.exit(fail === 0 ? 0 : 1)
