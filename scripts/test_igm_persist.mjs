// Verify IgmStore persistence: write facts, check the file, reload, confirm.
import { IgmStore } from '../dsh-igm-memory/lib/index.js'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const file = path.join(os.tmpdir(), 'igm-persist-test.json')
try { fs.unlinkSync(file) } catch {}

const s1 = new IgmStore(file)
s1.add('我的住址是北京。')
s1.add('我的住址现在是深圳了。')
console.log('after add: stored=%d, file exists=%s', s1.size, fs.existsSync(file))
console.log('file content:', fs.readFileSync(file, 'utf8').slice(0, 200))

// Simulate a restart: new store instance loading the same file.
const s2 = new IgmStore(file)
console.log('after reload: stored=%d', s2.size)
const hasOld = s2.items.some((i) => i.text.includes('北京'))
const hasNew = s2.items.some((i) => i.text.includes('深圳'))
console.log('reloaded: old_present=%s new_present=%s', hasOld, hasNew)
console.log(hasNew && !hasOld && s2.size === 1 ? 'PERSIST + SUPERSEDE OK' : 'FAIL')
try { fs.unlinkSync(file) } catch {}
