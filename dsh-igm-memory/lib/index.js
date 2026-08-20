// dsh-igm-memory: importance-gated memory write layer for DeepSeek Harness.
//
// A Host-side plugin that guards what the agent writes into its durable
// memory/instructions:
//   1. Gate: only memories above an importance threshold are written
//      (questions and filler are rejected).
//   2. Slot supersede: a fact about the same attribute replaces the old
//      value, so memories never accumulate stale/contradictory entries.
//
// This is a pure-JS port of the IGM mechanism (see igm/ in the repo root).
// Zero runtime dependencies; registered through a standard DSH bundle layer.

export const name = 'dsh-igm-memory'

const FACT_MARKERS = ['我', '我的', '喜欢', '是', '在', '去过', '住', '工作', '现在', '叫', '名字', '来自', '出生', '毕业', '擅长',
  // project-scope facts (code development): "这个项目用 pnpm", "项目采用 X"
  '这个项目', '项目用', '项目是', '项目采用', '项目', '仓库', '代码', '依赖', '技术栈', '构建', '部署', '约定', '架构',
  // high-value memory types: decisions, gotchas, root causes
  '因为', '所以', '原因', '选择', '选了', '踩过', '坑', '坑是', '注意', '记住', '下次', '别再', '当时', '决定', '方案']
const QUESTION_MARKERS = ['什么', '吗', '？', '?', '哪', '怎么', '如何', '为什么']
const EXPERIENCE_MARKERS = ['踩过', '坑', '根因', '教训', '下次', '别再', '曾经失败', '修复后', '解决办法', '注意事项']
const DECISION_MARKERS = ['决定', '选择', '选了', '方案', '采用']
const SLOT_PREFIXES = ['现在的', '目前的', '新的', '原来的', '以前的', '当前的']
const SLOT_ANCHORS = ['我的', '我']
const SLOT_STOPS = ['现在是', '是什么', '是', '了', '。', '，', ',', '？', '?']

// Topic dictionary: maps keywords to canonical topics so experiences can be
// matched ACROSS projects ("electron packaging gotcha" learned in project A
// surfaces when project B works on electron).
const TOPIC_KEYWORDS = {
  'electron': ['electron', '桌面端', '主进程', '渲染进程'],
  'packaging': ['打包', '构建', 'build', 'package', '安装器', '安装包'],
  'package-manager': ['npm', 'pnpm', 'yarn', '包管理器', 'lockfile', 'package-lock', 'packageManager'],
  'deploy': ['部署', 'deploy', '发布', 'release', '上线', 'ci', '流水线'],
  'database': ['数据库', 'db', 'mysql', 'postgres', 'sqlite', 'redis', 'mongo'],
  'backend': ['后端', 'server', 'api', '接口', 'express', 'koa', 'fastapi', 'flask'],
  'frontend': ['前端', 'react', 'vue', 'component', '组件', 'css', 'tailwind', 'ui'],
  'testing': ['测试', 'test', 'jest', 'vitest', 'pytest', '单测'],
  'docker': ['docker', '容器', '镜像', 'k8s', 'kubernetes'],
  'node': ['node', 'nodejs', 'vite', 'webpack', 'esbuild'],
  'auth': ['登录', '鉴权', 'auth', 'token', 'session', 'jwt', 'oauth'],
  'network': ['网络', '请求', 'http', 'https', '超时', '重试', 'proxy'],
  'oss-storage': ['oss', '对象存储', '存储桶', 'bucket', 's3'],
  'llm': ['llm', '模型', 'prompt', '推理', '生成', 'token', '上下文'],
}

export function extractTopics(text) {
  const found = []
  for (const [topic, keywords] of Object.entries(TOPIC_KEYWORDS)) {
    if (keywords.some((kw) => text.toLowerCase().includes(kw))) found.push(topic)
  }
  return found
}

export function extractSlot(text, maxLen = 6) {
  for (const anchor of SLOT_ANCHORS) {
    const ai = text.indexOf(anchor)
    if (ai < 0) continue
    const rest = text.slice(ai + anchor.length)
    for (const stop of SLOT_STOPS) {
      const si = rest.indexOf(stop)
      if (si >= 0) {
        let attr = rest.slice(0, si).trim()
        for (const pref of SLOT_PREFIXES) {
          if (attr.startsWith(pref)) attr = attr.slice(pref.length)
        }
        if (attr.length > 0 && attr.length <= maxLen) return attr
      }
    }
  }
  return null
}

function factScore(text) {
  let hits = 0
  for (const mk of FACT_MARKERS) if (text.includes(mk)) hits++
  return Math.min(hits / 3, 1)
}

function isQuestion(text) {
  return QUESTION_MARKERS.some((q) => text.includes(q))
}

function infoDensity(text) {
  return Math.min(text.split(/\s+/).length / 20, 1)
}

// HeuristicScorer port: questions are never memories; durable
// self-referential facts score high; filler stays below threshold.
export function importanceScore(text, maxSimToStore = 0) {
  if (isQuestion(text)) return 0
  const f = factScore(text)
  const surprise = 1 - maxSimToStore
  const density = infoDensity(text)
  return 0.6 * f + 0.3 * surprise + 0.1 * density
}

export function inferMemoryType(text) {
  if (EXPERIENCE_MARKERS.some((marker) => text.includes(marker))) return 'experience'
  if (DECISION_MARKERS.some((marker) => text.includes(marker))) return 'decision'
  return 'fact'
}

function normalizeMemoryType(type, text) {
  return ['fact', 'decision', 'experience'].includes(type) ? type : inferMemoryType(text)
}

// A minimal slot-aware memory store with JSON-file persistence.
// Persistence lives on the store so the shape stays swappable; the file path
// is injected by the plugin (defaults under $DSH_HOME/storages).
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import crypto from 'node:crypto'

export class IgmStore {
  constructor(filePath = null, defaults = {}) {
    this.items = []
    this.file = filePath
    this.defaults = defaults
    if (filePath) this.load()
  }

  load() {
    try {
      const raw = fs.readFileSync(this.file, 'utf8')
      const data = JSON.parse(raw)
      if (Array.isArray(data.items)) {
        // Older files remain readable; missing metadata is derived in memory and
        // is persisted on the next mutation.
        this.items = data.items
          .filter((it) => it && typeof it.text === 'string')
          .map((it) => this.normalizeItem(it))
      }
    } catch {
      this.items = [] // missing/corrupt file -> start fresh
    }
  }

  normalizeItem(item) {
    const text = item.text
    const fallbackScope = this.defaults.scope || (isProjectFact(text) ? 'project' : 'user')
    return {
      ...item,
      slot: typeof item.slot === 'string' ? item.slot : null,
      score: Number.isFinite(item.score) ? item.score : 0,
      ts: Number.isFinite(item.ts) ? item.ts : 0,
      reuseCount: Number.isFinite(item.reuseCount) ? item.reuseCount : 0,
      lastUsedAt: Number.isFinite(item.lastUsedAt) ? item.lastUsedAt : 0,
      type: normalizeMemoryType(item.type, text),
      scope: ['user', 'project'].includes(item.scope) ? item.scope : fallbackScope,
      projectId: item.projectId || this.defaults.projectId || null,
      cwd: item.cwd || this.defaults.cwd || null,
      topics: Array.isArray(item.topics) && item.topics.length ? item.topics : extractTopics(text),
      provenance: item.provenance || 'legacy',
    }
  }

  save() {
    if (!this.file) return
    try {
      fs.mkdirSync(path.dirname(this.file), { recursive: true })
      const tmp = this.file + '.tmp'
      fs.writeFileSync(tmp, JSON.stringify({ items: this.items }, null, 2))
      fs.renameSync(tmp, this.file)
    } catch (e) {
      console.log(`[igm-memory] persist failed: ${e.message}`)
    }
  }

  add(text, threshold = 0.6, maxSlotLen = 6, maxFactLen = 200, metadata = {}) {
    if (!text || typeof text !== 'string') return { kept: false, reason: 'invalid', score: 0 }
    if (text.length > maxFactLen) return { kept: false, reason: 'too_long', score: 0 }
    const score = importanceScore(text)
    if (score < threshold) return { kept: false, reason: 'gate', score }
    let slot = extractSlot(text, maxSlotLen)
    if (slot === null && isProjectFact(text)) slot = extractProjectSlot(text)
    const topics = extractTopics(text)
    const now = Date.now()
    const itemMetadata = {
      type: normalizeMemoryType(metadata.type, text),
      scope: metadata.scope || this.defaults.scope || (isProjectFact(text) ? 'project' : 'user'),
      projectId: metadata.projectId || this.defaults.projectId || null,
      cwd: metadata.cwd || this.defaults.cwd || null,
      provenance: metadata.provenance || 'service',
    }
    // Text-level dedup: identical or near-identical facts (even without a
    // slot) update the existing entry instead of appending a duplicate.
    const norm = text.replace(/\s+/g, '')
    const existing = this.items.find((it) => it.slot !== null && it.slot === slot)
      || this.items.find((it) => it.text.replace(/\s+/g, '') === norm)
    if (existing) {
      existing.text = text
      existing.score = score
      existing.ts = now
      existing.topics = topics.length ? topics : existing.topics
      Object.assign(existing, itemMetadata)
      this.save()
      return { kept: true, item: existing, score, deduped: true }
    }
    if (slot !== null) {
      this.items = this.items.filter((it) => it.slot !== slot) // supersede
    }
    const item = { text, slot, score, ts: now, reuseCount: 0, lastUsedAt: 0, topics, ...itemMetadata }
    this.items.push(item)
    this.save()
    return { kept: true, item, score }
  }

  query(text, topK = 3, maxSlotLen = 6) {
    if (typeof text !== 'string' || topK <= 0) return []
    const slot = extractSlot(text, maxSlotLen)
    const scored = this.items.map((item) => {
      let s = item.score
      if (slot !== null && item.slot === slot) s += 1 // slot-aware routing boost
      return { item, s }
    })
    scored.sort((a, b) => b.s - a.s)
    const top = scored.slice(0, topK)
    this.touch(top.map(({ item }) => item))
    return top.map(({ item, s }) => ({ ...item, s }))
  }

  // Retrieval = use: update only items owned by this store and persist once.
  touch(items = this.items) {
    const owned = new Set(this.items)
    const now = Date.now()
    let touched = 0
    for (const item of items) {
      if (!owned.has(item)) continue
      item.reuseCount = (item.reuseCount || 0) + 1
      item.lastUsedAt = now
      touched++
    }
    if (touched > 0) this.save()
    return touched
  }

  // Consolidation uses the most recent write/recall as activity. Stale items
  // below minScore fade out; a recent recall refreshes their retention window.
  consolidate(maxAgeDays = 30, minScore = 1) {
    const now = Date.now()
    const cutoff = now - maxAgeDays * 24 * 3600 * 1000
    const before = this.items.length
    this.items = this.items.filter((it) => {
      const lastActivity = Math.max(it.ts || 0, it.lastUsedAt || 0)
      if (lastActivity >= cutoff) return true // recently written or recalled
      return (it.score || 0) >= minScore      // only exceptional stale items remain
    })
    const removed = before - this.items.length
    if (removed > 0) this.save()
    return removed
  }

  get size() {
    return this.items.length
  }
}

// ---------------------------------------------------------------- dsh plugin
import z from 'schemastery'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const Config = z.object({
  enabled: z.boolean().default(true),
  writeThreshold: z.number().default(0.6),
  slotMaxLen: z.number().default(6),
  maxFactLen: z.number().default(200),
  maxInjectionBytes: z.number().default(2048),
  // Optional explicit store file for user facts. Profiles can set this in
  // their patch layer to isolate memories per profile (e.g. web vs headless).
  storeFile: z.string().default(''),
})

export const inject = ['tools', 'systemPrompt'] // model-facing tool + prompt injection

const TOOL_NAME = 'remember_fact'
const TOOL_DESCRIPTION =
  'Store a durable fact, decision, or reusable experience about the user or current project. ' +
  'Facts about the same attribute are replaced by the newest value. Questions and chit-chat are rejected. ' +
  'Use memoryType=experience only for reusable lessons or root causes; only experiences may transfer across projects.'

const RECALL_NAME = 'recall_fact'
const RECALL_DESCRIPTION =
  'Retrieve the durable facts stored about the user or this project (preferences, decisions, conventions, gotchas). ' +
  'Call this BEFORE answering when the user asks about something that may have been stated in a previous session, ' +
  'or when you are about to rely on a preference/convention. When you use a fact from memory, tell the user its source ' +
  '(e.g. "根据你之前说的..." / "按项目约定，之前记过..."). Returns current values only.'

// Project-scope markers: facts about the codebase/conventions live in the
// per-project store; everything else (user facts) lives in the shared store.
const PROJECT_MARKERS = ['这个项目', '项目用', '项目是', '项目采用', '项目', '仓库', '代码', '依赖', '技术栈', '构建', '部署', '约定', '架构', '前端', '后端', '数据库', '接口', '路由', '组件', '测试', 'CI', '发布']

function isProjectFact(text) {
  return PROJECT_MARKERS.some((m) => text.includes(m))
}

// Fallback slot for facts that are not "我的X是Y" shaped. Project facts like
// "这个项目使用 pnpm 作为包管理器" extract "包管理器" as their attribute.
function extractProjectSlot(text) {
  const m = text.match(/作为([\u4e00-\u9fa5A-Za-z0-9]{2,8})/)
  if (m) return m[1]
  const m2 = text.match(/项目(?:使用|采用|用|是)([\u4e00-\u9fa5A-Za-z0-9]{2,10})/)
  if (m2) return m2[1]
  return null
}

function cwdFromSession(session) {
  return session?.header?.cwd || session?.cwd || session?.meta?.cwd || null
}

function memoryView(item) {
  return {
    text: item.text,
    slot: item.slot || '',
    type: item.type,
    scope: item.scope,
    projectId: item.projectId || '',
    reuseCount: item.reuseCount || 0,
  }
}

export function apply(ctx, config) {
  const enabled = config.enabled ?? true
  const threshold = config.writeThreshold ?? 0.6
  const slotMaxLen = config.slotMaxLen ?? 6
  const maxFactLen = config.maxFactLen ?? 200
  const maxInjectionBytes = config.maxInjectionBytes ?? 2048
  const dshHome = process.env.DSH_HOME || path.join(os.homedir(), '.dsh')
  const storageDir = path.join(dshHome, 'storages')
  const configuredStoreFile = typeof config.storeFile === 'string' ? config.storeFile.trim() : ''
  const userStoreFile = configuredStoreFile ? path.resolve(configuredStoreFile) : path.join(storageDir, 'igm-user.json')
  const registryFile = path.join(storageDir, 'igm-projects.json')

  const stores = new Map() // project id -> { cwd, store }
  const projectRegistry = new Map()
  let sharedStore = null
  let unknownProjectStore = null

  const log = (msg) => {
    // console.log is used deliberately: ctx.logger is not an injected service.
    console.log(`[igm-memory] ${msg}`)
  }

  const canonicalCwd = (cwd) => {
    if (typeof cwd !== 'string' || !cwd.trim()) return null
    return path.normalize(path.resolve(cwd))
  }

  const projectIdFor = (cwd) => crypto.createHash('sha256').update(cwd).digest('hex').slice(0, 12)
  const projectStorePath = (projectId) => path.join(storageDir, `igm-project-${projectId}.json`)

  const loadRegistry = () => {
    try {
      const data = JSON.parse(fs.readFileSync(registryFile, 'utf8'))
      for (const [projectId, entry] of Object.entries(data.projects || {})) {
        if (/^[a-f0-9]{12}$/.test(projectId) && typeof entry?.cwd === 'string') {
          projectRegistry.set(projectId, { cwd: canonicalCwd(entry.cwd), updatedAt: entry.updatedAt || 0 })
        }
      }
    } catch {
      // The registry is an index only. Project files remain independently readable.
    }
  }

  const saveRegistry = () => {
    try {
      fs.mkdirSync(storageDir, { recursive: true })
      const projects = Object.fromEntries(projectRegistry)
      const tmp = registryFile + '.tmp'
      fs.writeFileSync(tmp, JSON.stringify({ version: 1, projects }, null, 2))
      fs.renameSync(tmp, registryFile)
    } catch (e) {
      log(`project registry persist failed: ${e.message}`)
    }
  }

  loadRegistry()

  const userStore = () => sharedStore || (sharedStore = new IgmStore(userStoreFile, { scope: 'user' }))

  const projectStore = (rawCwd) => {
    const cwd = canonicalCwd(rawCwd)
    if (!cwd) {
      if (!unknownProjectStore) {
        unknownProjectStore = new IgmStore(path.join(storageDir, 'igm-project-unknown.json'), {
          scope: 'project',
          projectId: 'unknown',
        })
      }
      return unknownProjectStore
    }

    const projectId = projectIdFor(cwd)
    if (!stores.has(projectId)) {
      stores.set(projectId, {
        cwd,
        store: new IgmStore(projectStorePath(projectId), { scope: 'project', projectId, cwd }),
      })
    }
    if (projectRegistry.get(projectId)?.cwd !== cwd) {
      projectRegistry.set(projectId, { cwd, updatedAt: Date.now() })
      saveRegistry()
    }
    return stores.get(projectId).store
  }

  // Discover persisted project stores on every cross-project lookup. The
  // registry restores their cwd labels; orphaned files still work by hash.
  const allProjectStores = () => {
    try {
      for (const filename of fs.readdirSync(storageDir)) {
        const match = filename.match(/^igm-project-([a-f0-9]{12})\.json$/)
        if (!match || stores.has(match[1])) continue
        const projectId = match[1]
        const cwd = projectRegistry.get(projectId)?.cwd || null
        stores.set(projectId, {
          cwd,
          store: new IgmStore(path.join(storageDir, filename), { scope: 'project', projectId, cwd }),
        })
      }
    } catch {
      // No storage directory yet.
    }
    return [...stores.entries()]
      .map(([projectId, entry]) => ({ projectId, ...entry }))
      .filter(({ store }) => store.size > 0)
  }

  const crossProjectExperiences = (topics, excludeCwd, limit = 4) => {
    if (!Array.isArray(topics) || topics.length === 0) return []
    const excludeId = canonicalCwd(excludeCwd) ? projectIdFor(canonicalCwd(excludeCwd)) : null
    const hits = []
    for (const { projectId, cwd, store } of allProjectStores()) {
      if (projectId === excludeId) continue
      for (const item of store.items) {
        const itemTopics = item.topics || []
        if (item.scope === 'project' && item.type === 'experience' && itemTopics.some((topic) => topics.includes(topic))) {
          hits.push({ text: item.text, topics: itemTopics, project: cwd || `project:${projectId}`, ts: item.ts, item, store })
        }
      }
    }
    hits.sort((a, b) => (b.ts || 0) - (a.ts || 0))
    return hits.slice(0, limit)
  }

  const addMemory = (text, cwd, memoryType, provenance) => {
    const scope = typeof text === 'string' && isProjectFact(text) ? 'project' : 'user'
    const canonical = canonicalCwd(cwd)
    const projectId = scope === 'project' && canonical ? projectIdFor(canonical) : null
    const store = scope === 'project' ? projectStore(canonical) : userStore()
    return store.add(text, threshold, slotMaxLen, maxFactLen, {
      type: memoryType,
      scope,
      projectId,
      cwd: scope === 'project' ? canonical : null,
      provenance,
    })
  }

  if (!enabled) {
    log('disabled by config')
    return
  }

  log(`enabled (threshold=${threshold}, slotMaxLen=${slotMaxLen}, auto-scope routing)`)
  log(`user store: ${userStore().file} (${userStore().size} persisted)`)

  // Services require an explicit cwd for project-scoped operations. Omitting it
  // deliberately routes project facts to the isolated unknown-project store.
  ctx.provide('igm.memory.write', (text, options = {}) => {
    const res = addMemory(text, options.cwd, options.type, options.provenance || 'service')
    log(res.kept ? `kept [${res.item.type}/${res.item.slot || 'none'}] ${text.slice(0, 40)}` : `filtered: ${String(text).slice(0, 40)}`)
    return res
  })

  ctx.provide('igm.memory.query', (text, cwd = null) => {
    const all = [...userStore().query(text, 3, slotMaxLen), ...projectStore(cwd).query(text, 3, slotMaxLen)]
    return all.sort((a, b) => b.s - a.s).slice(0, 3)
  })

  ctx.provide('igm.memory.stats', (cwd = null) => {
    const canonical = canonicalCwd(cwd)
    const u = userStore()
    const p = projectStore(canonical)
    return {
      stored: u.size + p.size,
      cwd: canonical,
      userItems: u.items.map(memoryView),
      projectItems: p.items.map(memoryView),
    }
  })

  ctx.provide('igm.memory.list', (cwd = null) => {
    const u = userStore()
    const p = projectStore(cwd)
    return { user: u.items.map(memoryView), project: p.items.map(memoryView) }
  })

  ctx.provide('igm.memory.consolidate', (maxAgeDays = 30, minScore = 1, cwd = null) => {
    const u = userStore()
    const p = projectStore(cwd)
    const removed = u.consolidate(maxAgeDays, minScore) + p.consolidate(maxAgeDays, minScore)
    log(`consolidate removed ${removed} (${u.size + p.size} remain)`)
    return { removed, remaining: u.size + p.size }
  })

  ctx.tools.register(defineTool({
    name: TOOL_NAME,
    description: TOOL_DESCRIPTION,
    parameters: {
      fact: {
        type: 'string',
        required: true,
        description: 'The durable memory, phrased as a user or current-project statement.',
      },
      memoryType: {
        type: 'string',
        description: 'Optional classification: fact, decision, or experience. Omit to infer it.',
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: true,
        properties: {
          stored: { type: 'boolean' },
          slot: { type: 'string' },
          reason: { type: 'string' },
          memory: { type: 'array', items: { type: 'object', additionalProperties: true } },
        },
      },
      render(args, value) {
        return [{ type: 'text', text: JSON.stringify(value) }]
      },
    },
    async execute(args, exec) {
      const cwd = cwdFromSession(exec?.agent?.session)
      const res = addMemory(args.fact, cwd, args.memoryType, 'remember_fact')
      const memory = [...userStore().items, ...projectStore(cwd).items].map(memoryView)
      if (res.kept) {
        log(`tool kept [${res.item.type}/${res.item.slot || 'none'}] ${args.fact.slice(0, 50)}`)
        return { stored: true, slot: res.item.slot || '', reason: 'stored', memory }
      }
      log(`tool filtered: ${args.fact.slice(0, 50)}`)
      return { stored: false, slot: '', reason: res.reason, memory }
    },
  }))
  log(`tool registered: ${TOOL_NAME}`)

  ctx.tools.register(defineTool({
    name: RECALL_NAME,
    description: RECALL_DESCRIPTION,
    parameters: {},
    output: {
      schema: {
        type: 'object',
        additionalProperties: true,
        properties: {
          memory: { type: 'array', items: { type: 'object', additionalProperties: true } },
          experiences: { type: 'array', items: { type: 'object', additionalProperties: true } },
        },
      },
      render(args, value) {
        return [{ type: 'text', text: JSON.stringify(value) }]
      },
    },
    async execute(args, exec) {
      const cwd = cwdFromSession(exec?.agent?.session)
      const u = userStore()
      const p = projectStore(cwd)
      const ownTopics = new Set(p.items.flatMap((item) => item.topics || []))
      const experienceHits = crossProjectExperiences([...ownTopics], cwd, 4)
      // recall_fact is the model-facing retrieval path, so it must refresh
      // retention just like igm.memory.query does.
      u.touch()
      p.touch()
      const experienceItemsByStore = new Map()
      for (const hit of experienceHits) {
        const items = experienceItemsByStore.get(hit.store) || []
        items.push(hit.item)
        experienceItemsByStore.set(hit.store, items)
      }
      for (const [store, items] of experienceItemsByStore) store.touch(items)
      const memory = [...u.items, ...p.items].map(memoryView)
      const experiences = experienceHits
        .map((item) => ({ text: item.text, project: item.project, type: 'experience' }))
      log(`recall -> ${memory.length} memories (${u.size} user, ${p.size} project) + ${experiences.length} cross-project experiences`)
      return { memory, experiences }
    },
  }))
  log(`tool registered: ${RECALL_NAME}`)

  ctx.on('system-prompt/assemble', async (assembly, context, next) => {
    const assembled = await next()
    if (!enabled) return assembled
    const cwd = cwdFromSession(context?.agent?.session)
    if (cwd) log(`session cwd: ${cwd}`)
    const sectionName = 'igm-memory'
    const sections = Array.isArray(assembled?.sections) ? assembled.sections : []
    const filtered = sections.filter((section) => section?.name !== sectionName)
    const u = userStore()
    const p = projectStore(cwd)
    const ordered = [...u.items, ...p.items].sort((a, b) => (b.ts || 0) - (a.ts || 0))
    const lines = []
    let budget = maxInjectionBytes
    for (const item of ordered) {
      const line = `- [${item.scope}/${item.type}] ${item.text}`
      const bytes = Buffer.byteLength(line, 'utf8')
      if (bytes > budget) continue
      lines.push(line)
      budget -= bytes
    }

    const parts = [
      'IGM memory rules (follow proactively, do not wait for the user to say "remember"):\n' +
      '1. Store durable user facts with remember_fact; phrase them as "我的{attr}是{value}".\n' +
      '2. Store durable facts about THIS project with "这个项目{...}" so they remain project-scoped.\n' +
      '3. Classify stable state as fact, a chosen approach as decision, and a reusable lesson/root cause as experience.\n' +
      '4. When a known value changes, store the new value so slot supersede removes the old one.\n' +
      '5. When answering from memory, cite it as previously stated or as a project convention.\n' +
      '6. Do not store questions, chit-chat, secrets, or one-off requests.',
    ]
    if (lines.length > 0) {
      parts.push('Durable memories from previous sessions ([scope/type]; current values only):\n' + lines.join('\n'))
    }

    const ownTopics = new Set(p.items.flatMap((item) => item.topics || []))
    const experienceLines = []
    for (const experience of crossProjectExperiences([...ownTopics], cwd, 3)) {
      const project = experience.project.replace(/\\/g, '/').split('/').filter(Boolean).slice(-2).join('/')
      const line = `- [experience from ${project}] ${experience.text}`
      const bytes = Buffer.byteLength(line, 'utf8')
      if (bytes > budget) continue
      experienceLines.push(line)
      budget -= bytes
    }
    if (experienceLines.length > 0) {
      parts.push('Relevant experiences from other projects (topic-matched lessons only):\n' + experienceLines.join('\n'))
    }

    filtered.push({ name: sectionName, text: parts.join('\n\n'), order: 5 })
    return { ...assembled, sections: filtered }
  })
  log(`system-prompt injection armed (memory budget ${maxInjectionBytes}B)`)
  log('services registered: igm.memory.write / query / stats / list / consolidate')
}
