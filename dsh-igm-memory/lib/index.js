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
const SLOT_PREFIXES = ['现在的', '目前的', '新的', '原来的', '以前的', '当前的']
const SLOT_ANCHORS = ['我的', '我']
const SLOT_STOPS = ['现在是', '是什么', '是', '了', '。', '，', ',', '？', '?']

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

// A minimal slot-aware memory store with JSON-file persistence.
// Persistence lives on the store so the shape stays swappable; the file path
// is injected by the plugin (defaults under $DSH_HOME/storages).
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import crypto from 'node:crypto'

export class IgmStore {
  constructor(filePath = null) {
    this.items = [] // { text, slot, score, ts }
    this.file = filePath
    if (filePath) this.load()
  }

  load() {
    try {
      const raw = fs.readFileSync(this.file, 'utf8')
      const data = JSON.parse(raw)
      if (Array.isArray(data.items)) this.items = data.items
    } catch {
      this.items = [] // missing/corrupt file -> start fresh
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

  add(text, threshold = 0.6, maxSlotLen = 6, maxFactLen = 200) {
    if (!text || typeof text !== 'string') return { kept: false, reason: 'invalid', score: 0 }
    if (text.length > maxFactLen) return { kept: false, reason: 'too_long', score: 0 }
    const score = importanceScore(text)
    if (score < threshold) return { kept: false, reason: 'gate', score }
    let slot = extractSlot(text, maxSlotLen)
    if (slot === null && isProjectFact(text)) slot = extractProjectSlot(text)
    // Text-level dedup: identical or near-identical facts (even without a
    // slot) update the existing entry instead of appending a duplicate.
    const norm = text.replace(/\s+/g, '')
    const existing = this.items.find((it) => it.slot !== null && it.slot === slot)
      || this.items.find((it) => it.text.replace(/\s+/g, '') === norm)
    if (existing) {
      existing.text = text
      existing.score = score
      existing.ts = Date.now()
      this.save()
      return { kept: true, item: existing, score, deduped: true }
    }
    if (slot !== null) {
      this.items = this.items.filter((it) => it.slot !== slot) // supersede
    }
    const item = { text, slot, score, ts: Date.now(), reuseCount: 0 }
    this.items.push(item)
    this.save()
    return { kept: true, item, score }
  }

  query(text, topK = 3, maxSlotLen = 6) {
    const slot = extractSlot(text, maxSlotLen)
    const scored = this.items.map((it) => {
      let s = it.score
      if (slot !== null && it.slot === slot) s += 1 // slot-aware routing boost
      return { ...it, s }
    })
    scored.sort((a, b) => b.s - a.s)
    const top = scored.slice(0, topK)
    // Retrieval = use: bump reuse so consolidation keeps recently-used facts.
    for (const hit of top) {
      const orig = this.items.find((it) => it === hit)
      if (orig) orig.reuseCount = (orig.reuseCount || 0) + 1
    }
    return top
  }

  // Consolidation: drop memories that are old AND were never reused.
  // Older entries that were recalled stay; stale unused ones fade out.
  consolidate(maxAgeDays = 30, minScore = 0.5) {
    const now = Date.now()
    const cutoff = now - maxAgeDays * 24 * 3600 * 1000
    const before = this.items.length
    this.items = this.items.filter((it) => {
      const ts = it.ts || 0
      if (ts >= cutoff) return true        // recent: keep
      if (it.reuseCount > 0) return true   // reused: keep
      return (it.score || 0) >= minScore   // old + never used + low value: drop
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
  'Store a durable fact about the user or project (e.g. "我的住址是北京"). ' +
  'Call this when the user states a persistent preference, address, role, or decision. ' +
  'Facts about the SAME attribute are automatically replaced by the newest value, ' +
  'so stale/contradictory entries never accumulate. Non-facts (questions, chit-chat) are rejected.'

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

// Full slot resolution: user facts use 我的X是Y; project facts fall back to
// the project-slot extractor so updates still supersede by attribute.
function resolveSlot(text, maxLen) {
  const userSlot = extractSlot(text, maxLen)
  if (userSlot) return userSlot
  if (isProjectFact(text)) return extractProjectSlot(text)
  return null
}

export function apply(ctx, config) {
  const enabled = config.enabled ?? true
  const threshold = config.writeThreshold ?? 0.6
  const slotMaxLen = config.slotMaxLen ?? 6
  const maxFactLen = config.maxFactLen ?? 200
  const maxInjectionBytes = config.maxInjectionBytes ?? 2048
  const dshHome = process.env.DSH_HOME || path.join(os.homedir(), '.dsh')

  // Store manager: project facts auto-route to a per-cwd store; user facts
  // always go to the shared store. No configuration needed.
  const stores = new Map()          // cwd -> IgmStore (project facts)
  let sharedStore = null            // user facts (all projects)
  let unknownProjectStore = null    // project facts with unknown cwd
  let currentCwd = null             // cwd of the most recent assembled session

  const projectStorePath = (cwd) => {
    const hash = crypto.createHash('sha256').update(cwd).digest('hex').slice(0, 12)
    return path.join(dshHome, 'storages', `igm-project-${hash}.json`)
  }

  const userStore = () => sharedStore || (sharedStore = new IgmStore(path.join(dshHome, 'storages', 'igm-user.json')))

  const projectStore = (cwd) => {
    if (!cwd) cwd = currentCwd
    if (!cwd) {
      // Project fact but no session cwd yet: keep it OUT of the user store.
      return unknownProjectStore || (unknownProjectStore = new IgmStore(path.join(dshHome, 'storages', 'igm-project-unknown.json')))
    }
    let s = stores.get(cwd)
    if (!s) {
      s = new IgmStore(projectStorePath(cwd))
      stores.set(cwd, s)
    }
    return s
  }

  // Route a candidate fact to the right store by its scope.
  const storeFor = (text, cwd) => {
    if (isProjectFact(text)) return projectStore(cwd)
    return userStore()
  }

  const log = (msg) => {
    // console.log is used deliberately: ctx.logger may not be injectable
    // without declaring it, and we must not risk another load failure.
    console.log(`[igm-memory] ${msg}`)
  }

  if (!enabled) {
    log('disabled by config')
    return
  }

  log(`enabled (threshold=${threshold}, slotMaxLen=${slotMaxLen}, auto-scope routing)`)
  log(`user store: ${userStore().file} (${userStore().size} persisted)`)

  // Memory-write guard: expose a service so other plugins / the agent loop
  // can route memory candidates through the IGM gate.
  ctx.provide('igm.memory.write', (text) => {
    const store = storeFor(text, currentCwd)
    const res = store.add(text, threshold, slotMaxLen, maxFactLen)
    log(res.kept ? `kept [${res.item.slot || 'none'}] ${text.slice(0, 40)}` : `filtered: ${text.slice(0, 40)}`)
    return res
  })

  ctx.provide('igm.memory.query', (text) => {
    // Query both scopes; slot-aware ranking decides.
    const all = [...userStore().query(text, 3, slotMaxLen), ...projectStore(currentCwd).query(text, 3, slotMaxLen)]
    return all.slice(0, 3)
  })

  ctx.provide('igm.memory.stats', () => {
    const u = userStore()
    const p = projectStore(currentCwd)
    return {
      stored: u.size + p.size,
      cwd: currentCwd,
      userItems: u.items.map((it) => ({ text: it.text, slot: it.slot, reuseCount: it.reuseCount || 0 })),
      projectItems: p.items.map((it) => ({ text: it.text, slot: it.slot, reuseCount: it.reuseCount || 0 })),
    }
  })

  // Full listing with scope labels — lets the user inspect what is remembered.
  ctx.provide('igm.memory.list', () => {
    const u = userStore()
    const p = projectStore(currentCwd)
    return {
      user: u.items.map((it) => ({ text: it.text, slot: it.slot || '' })),
      project: p.items.map((it) => ({ text: it.text, slot: it.slot || '' })),
    }
  })

  ctx.provide('igm.memory.consolidate', (maxAgeDays = 30, minScore = 0.5) => {
    const u = userStore()
    const p = projectStore(currentCwd)
    const removed = u.consolidate(maxAgeDays, minScore) + p.consolidate(maxAgeDays, minScore)
    log(`consolidate removed ${removed} (${u.size + p.size} remain)`)
    return { removed, remaining: u.size + p.size }
  })

  // Model-facing tool: lets the agent persist facts through the IGM gate.
  ctx.tools.register(defineTool({
    name: TOOL_NAME,
    description: TOOL_DESCRIPTION,
    parameters: {
      fact: {
        type: 'string',
        required: true,
        description: 'The fact to remember, phrased as a self-referential statement (e.g. "我的住址是北京" / "我的住址现在是深圳").',
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
          memory: {
            type: 'array',
            items: { type: 'object', additionalProperties: true },
          },
        },
      },
      render(args, value) {
        return [{ type: 'text', text: JSON.stringify(value) }]
      },
    },
    async execute(args) {
      const store = storeFor(args.fact, currentCwd)
      const res = store.add(args.fact, threshold, slotMaxLen, maxFactLen)
      const memory = [...userStore().items, ...projectStore(currentCwd).items].map((it) => ({ text: it.text, slot: it.slot || '' }))
      if (res.kept) {
        log(`tool kept [${res.item.slot || 'none'}] ${args.fact.slice(0, 50)}`)
        return {
          stored: true,
          slot: res.item.slot || '',
          reason: 'stored',
          memory,
        }
      }
      log(`tool filtered: ${args.fact.slice(0, 50)}`)
      return {
        stored: false,
        slot: '',
        reason: res.reason,
        memory,
      }
    },
  }))

  log(`tool registered: ${TOOL_NAME}`)

  // Model-facing read tool: lets the agent retrieve stored facts.
  ctx.tools.register(defineTool({
    name: RECALL_NAME,
    description: RECALL_DESCRIPTION,
    parameters: {},
    output: {
      schema: {
        type: 'object',
        additionalProperties: true,
        properties: {
          memory: {
            type: 'array',
            items: { type: 'object', additionalProperties: true },
          },
        },
      },
      render(args, value) {
        return [{ type: 'text', text: JSON.stringify(value) }]
      },
    },
    async execute() {
      // Retrieve both scopes so user + project facts are visible together.
      const u = userStore()
      const p = projectStore(currentCwd)
      const memory = [...u.items, ...p.items].map((it) => ({ text: it.text, slot: it.slot || '' }))
      log(`recall -> ${memory.length} facts (${u.size} user, ${p.size} project)`)
      return { memory }
    },
  }))
  log(`tool registered: ${RECALL_NAME}`)

  // Session-start injection: make persisted facts visible to the agent from
  // the first turn, AND teach it to persist facts proactively. The rules are
  // injected even with an empty store so a fresh profile still learns them.
  ctx.on('system-prompt/assemble', async (assembly, context, next) => {
    const assembled = await next()
    if (!enabled) return assembled
    // Resolve the session's working directory for per-project memory.
    const session = context?.agent?.session
    const cwd = session?.cwd || session?.meta?.cwd || null
    if (cwd) {
      currentCwd = cwd
      log(`session cwd: ${cwd}`)
    }
    const sectionName = 'igm-memory'
    const sections = Array.isArray(assembled?.sections) ? assembled.sections : []
    const filtered = sections.filter((s) => s?.name !== sectionName)
    // Build the fact list newest-first, cutting once the byte budget is hit,
    // so a large memory never blows up the agent's context window.
    const u = userStore()
    const p = projectStore(cwd)
    const ordered = [
      ...u.items.map((it) => ({ ...it, scope: 'user' })),
      ...p.items.map((it) => ({ ...it, scope: 'project' })),
    ].sort((a, b) => (b.ts || 0) - (a.ts || 0))
    const lines = []
    let budget = maxInjectionBytes
    for (const it of ordered) {
      const tag = it.scope === 'project' ? '[project]' : '[user]'
      const line = `- ${tag} ${it.text}`
      if (line.length > budget) break
      lines.push(line)
      budget -= line.length
    }
    const parts = []
    parts.push(
      'IGM memory rules (follow proactively, do not wait for the user to say "remember"):\n' +
      '1. When the user states any durable fact about themselves (address, preference, role, stack, tooling, etc.), immediately call remember_fact with "我的{attr}是{value}".\n' +
      '2. When the user states a durable fact about THIS project (stack, convention, decision, architecture), immediately call remember_fact with "这个项目{...}".\n' +
      '3. When the user explains WHY a decision was made or reveals a gotcha/bug root cause, remember it as a durable fact too — these are the most valuable memories.\n' +
      '4. When a previously known fact changes, immediately call remember_fact with the new value — the old value is automatically replaced.\n' +
      '5. When answering from memory, mention the source: e.g. "根据你之前说的..." or "按项目约定(之前记过)...". Do not silently pretend you always knew.\n' +
      '6. Do not store questions, chit-chat, or one-off requests; the gate rejects them anyway.'
    )
    if (lines.length > 0) {
      parts.push(
        'Durable facts remembered in previous sessions ([user]=你的偏好, [project]=本项目约定; only current values remain):\n' + lines.join('\n')
      )
    }
    filtered.push({
      name: sectionName,
      text: parts.join('\n\n'),
      order: 5,
    })
    return { ...assembled, sections: filtered }
  })
  log(`system-prompt injection armed (budget ${maxInjectionBytes}B)`)

  log('services registered: igm.memory.write / igm.memory.query / igm.memory.stats')
}
