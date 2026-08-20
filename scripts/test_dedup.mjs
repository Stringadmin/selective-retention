// Verify text-dedup: same fact twice -> one entry; changed value -> updates.
import { IgmStore } from '../dsh-igm-memory/lib/index.js'

let fail = 0
const ok = (cond, msg) => { console.log(`  ${cond ? 'ok' : 'FAIL'}  ${msg}`); if (!cond) fail++ }

const s = new IgmStore(null)
const r1 = s.add('这个项目使用 pnpm 作为包管理器')
const r2 = s.add('这个项目使用 pnpm 作为包管理器')
ok(r1.kept && r2.kept && r2.deduped, `same sentence deduped (size=${s.size})`)
ok(s.size === 1, 'no duplicate entries')

const r3 = s.add('这个项目使用 npm 作为包管理器')
ok(r3.kept && s.size === 1, `value changed, still 1 entry (size=${s.size})`)
ok(s.items[0].text.includes('npm'), 'text updated to npm')

console.log(fail === 0 ? '\nDEDUP OK' : `\n${fail} FAILURES`)
process.exit(fail === 0 ? 0 : 1)
