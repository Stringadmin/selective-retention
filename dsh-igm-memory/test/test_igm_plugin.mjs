import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const pluginRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const mod = await import(path.join(pluginRoot, 'lib/index.js'))
const { IgmStore, extractSlot, importanceScore, inferMemoryType } = mod

let pass = 0
const tempDirs = []

const test = async (name, fn) => {
  try {
    await fn()
    pass++
    console.log(`  ok  ${name}`)
  } catch (error) {
    console.error(`FAIL ${name}: ${error.stack || error.message}`)
    process.exitCode = 1
  }
}

const tempDir = () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'igm-test-'))
  tempDirs.push(dir)
  return dir
}

const execFor = (cwd) => ({ agent: { session: { header: { cwd } } } })

const createPlugin = (dshHome, rawConfig = {}) => {
  const tools = []
  const events = {}
  const services = new Map()
  const ctx = {
    provide(name, value) { services.set(name, value) },
    tools: { register(definition) { tools.push(definition) } },
    on(event, fn) { events[event] = fn },
    effect() {},
  }
  const originalHome = process.env.DSH_HOME
  process.env.DSH_HOME = dshHome
  try {
    mod.apply(ctx, mod.Config(rawConfig))
  } finally {
    if (originalHome === undefined) delete process.env.DSH_HOME
    else process.env.DSH_HOME = originalHome
  }
  return {
    events,
    services,
    remember: tools.find((tool) => tool.name === 'remember_fact'),
    recall: tools.find((tool) => tool.name === 'recall_fact'),
    toolNames: tools.map((tool) => tool.name).sort(),
  }
}

await test('slot extraction handles facts, updates, and questions', () => {
  assert.equal(extractSlot('我的住址是北京。'), '住址')
  assert.equal(extractSlot('更新一下，我的住址现在是深圳了。'), '住址')
  assert.equal(extractSlot('我现在的住址是什么？'), '住址')
  assert.equal(extractSlot('我的很长很长的属性名称是值。', 6), null)
  assert.equal(extractSlot('今天天气不错。'), null)
})

await test('importance gate keeps facts and rejects questions/filler', () => {
  assert.ok(importanceScore('我的职业是软件工程师。') >= 0.6)
  assert.equal(importanceScore('我的爱好是什么？'), 0)
  assert.ok(importanceScore('今天天气不错。') < 0.6)
})

await test('memory types distinguish facts, decisions, and experiences', () => {
  assert.equal(inferMemoryType('这个项目用 pnpm。'), 'fact')
  assert.equal(inferMemoryType('这个项目决定采用 SQLite。'), 'decision')
  assert.equal(inferMemoryType('这个项目踩过一个打包路径的坑。'), 'experience')
})

await test('slot supersede removes the old value', () => {
  const store = new IgmStore()
  store.add('我的住址是北京。')
  store.add('我的住址现在是深圳了。')
  assert.equal(store.size, 1)
  assert.match(store.items[0].text, /深圳/)
})

await test('identical project memories deduplicate and slot updates replace', () => {
  const store = new IgmStore()
  const first = store.add('这个项目使用 pnpm 作为包管理器。')
  const duplicate = store.add('这个项目使用 pnpm 作为包管理器。')
  assert.equal(first.kept, true)
  assert.equal(duplicate.deduped, true)
  assert.equal(store.size, 1)
  store.add('这个项目使用 bun 作为包管理器。')
  assert.equal(store.size, 1)
  assert.match(store.items[0].text, /bun/)
})

await test('store rejects invalid and oversized values', () => {
  const store = new IgmStore()
  assert.equal(store.add(null).reason, 'invalid')
  assert.equal(store.add('我的住址是' + '很长很长'.repeat(100) + '。', 0.6, 6, 200).reason, 'too_long')
})

await test('service boundary rejects invalid writes without throwing', () => {
  const plugin = createPlugin(tempDir())
  const result = plugin.services.get('igm.memory.write')(null)
  assert.equal(result.kept, false)
  assert.equal(result.reason, 'invalid')
})

await test('persistence survives restart and corrupt files are tolerated', () => {
  const dir = tempDir()
  const file = path.join(dir, 'memory.json')
  const first = new IgmStore(file)
  first.add('我的住址是北京。')
  first.add('我的住址现在是深圳了。')
  const restarted = new IgmStore(file)
  assert.equal(restarted.size, 1)
  assert.match(restarted.items[0].text, /深圳/)

  const corrupt = path.join(dir, 'corrupt.json')
  fs.writeFileSync(corrupt, '{not valid json')
  assert.equal(new IgmStore(corrupt).size, 0)
})

await test('legacy project items receive safe type and scope metadata', () => {
  const file = path.join(tempDir(), 'legacy-project.json')
  fs.writeFileSync(file, JSON.stringify({
    items: [{
      text: '这个项目用 npm。',
      slot: null,
      score: 0.9,
      ts: Date.now(),
      topics: ['package-manager'],
    }],
  }))
  const store = new IgmStore(file, { scope: 'project', projectId: 'legacy-id', cwd: '/legacy/project' })
  assert.equal(store.items[0].type, 'fact')
  assert.equal(store.items[0].scope, 'project')
  assert.equal(store.items[0].projectId, 'legacy-id')
})

await test('query persists reuse count and last-used time', () => {
  const file = path.join(tempDir(), 'memory.json')
  const store = new IgmStore(file)
  store.add('我的职业是软件工程师。')
  store.query('我的职业是什么？')
  const restarted = new IgmStore(file)
  assert.equal(restarted.items[0].reuseCount, 1)
  assert.ok(restarted.items[0].lastUsedAt > 0)
})

await test('default consolidation removes stale accepted memories', () => {
  const store = new IgmStore()
  const accepted = store.add('我的职业是软件工程师。')
  assert.equal(accepted.kept, true)
  store.items[0].ts = Date.now() - 60 * 24 * 3600 * 1000
  assert.equal(store.consolidate(), 1)
  assert.equal(store.size, 0)
})

await test('recent recall protects an otherwise stale memory', () => {
  const store = new IgmStore()
  store.add('我的职业是软件工程师。')
  store.items[0].ts = Date.now() - 60 * 24 * 3600 * 1000
  store.query('我的职业是什么？')
  assert.equal(store.consolidate(), 0)
  assert.equal(store.size, 1)
})

await test('tool calls route by their own session under interleaving', async () => {
  const home = tempDir()
  const projectA = path.join(home, 'project-a')
  const projectB = path.join(home, 'project-b')
  const plugin = createPlugin(home)
  assert.deepEqual(plugin.toolNames, ['recall_fact', 'remember_fact'])

  await plugin.events['system-prompt/assemble']({}, { agent: { session: { header: { cwd: projectB } } } }, async () => ({ sections: [] }))
  await Promise.all([
    plugin.remember.execute({ fact: '这个项目用 pnpm。' }, execFor(projectA)),
    plugin.remember.execute({ fact: '这个项目用 npm。' }, execFor(projectB)),
  ])

  const recalledA = await plugin.recall.execute({}, execFor(projectA))
  const recalledB = await plugin.recall.execute({}, execFor(projectB))
  const textA = recalledA.memory.map((item) => item.text).join(' | ')
  const textB = recalledB.memory.map((item) => item.text).join(' | ')
  assert.match(textA, /pnpm/)
  assert.doesNotMatch(textA, /用 npm。/)
  assert.match(textB, /用 npm。/)
  assert.doesNotMatch(textB, /pnpm/)
})

await test('cross-project recall survives restart and migrates only experiences', async () => {
  const home = tempDir()
  const projectA = path.join(home, 'project-a')
  const projectB = path.join(home, 'project-b')
  const first = createPlugin(home)
  await first.remember.execute({ fact: '这个项目踩过一个坑：electron 打包时 icon 路径要写绝对路径。' }, execFor(projectA))
  await first.remember.execute({ fact: '这个项目用 pnpm 作为包管理器。' }, execFor(projectA))
  await first.remember.execute({ fact: '这个项目用 electron，并使用 npm 作为包管理器。' }, execFor(projectB))

  const restarted = createPlugin(home)
  const recalled = await restarted.recall.execute({}, execFor(projectB))
  assert.equal(recalled.experiences.length, 1)
  assert.match(recalled.experiences[0].text, /icon 路径/)
  assert.ok(recalled.experiences.every((item) => !item.text.includes('pnpm')))
  assert.equal(recalled.memory[0].reuseCount, 1)
  const sourceStats = restarted.services.get('igm.memory.stats')(projectA)
  assert.equal(sourceStats.projectItems.find((item) => item.type === 'experience').reuseCount, 1)
  assert.ok(fs.existsSync(path.join(home, 'storages', 'igm-projects.json')))

  const assembled = await restarted.events['system-prompt/assemble'](
    {},
    { agent: { session: { header: { cwd: projectB } } } },
    async () => ({ sections: [] }),
  )
  const section = assembled.sections.find((item) => item.name === 'igm-memory')
  assert.match(section.text, /\[experience from .*project-a\]/)
  assert.doesNotMatch(section.text, /experience from .*pnpm/)
})

await test('storeFile controls the user-memory path', async () => {
  const home = tempDir()
  const customFile = path.join(home, 'profile-memory.json')
  const plugin = createPlugin(home, { storeFile: customFile })
  await plugin.remember.execute({ fact: '我的常用语言是 Python。' }, execFor(path.join(home, 'project')))
  assert.ok(fs.existsSync(customFile))
  assert.equal(fs.existsSync(path.join(home, 'storages', 'igm-user.json')), false)
  const data = JSON.parse(fs.readFileSync(customFile, 'utf8'))
  assert.equal(data.items[0].scope, 'user')
  assert.equal(data.items[0].provenance, 'remember_fact')
})

await test('injection carries typed memories within a UTF-8 byte budget', async () => {
  const home = tempDir()
  const cwd = path.join(home, 'project')
  const plugin = createPlugin(home, { maxInjectionBytes: 2048 })
  await plugin.remember.execute({ fact: '这个项目决定采用 pnpm。', memoryType: 'decision' }, execFor(cwd))
  const assembled = await plugin.events['system-prompt/assemble'](
    {},
    { agent: { session: { header: { cwd } } } },
    async () => ({ sections: [] }),
  )
  const section = assembled.sections.find((item) => item.name === 'igm-memory')
  assert.ok(section)
  assert.match(section.text, /\[project\/decision\]/)
  assert.match(section.text, /cite it as previously stated/)
})

for (const dir of tempDirs) fs.rmSync(dir, { recursive: true, force: true })

console.log(`\n${pass} tests passed` + (process.exitCode ? ' (with failures)' : ''))
process.exit(process.exitCode || 0)
