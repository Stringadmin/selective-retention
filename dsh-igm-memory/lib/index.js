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

const FACT_MARKERS = ['我', '我的', '喜欢', '是', '在', '去过', '住', '工作', '现在', '叫', '名字', '来自', '出生', '毕业', '擅长']
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

// A minimal slot-aware memory store (in-memory).  In a production plugin this
// would persist to disk; the shape is kept so the store can be swapped.
export class IgmStore {
  constructor() {
    this.items = [] // { text, slot, ts }
  }

  add(text, threshold = 0.6, maxSlotLen = 6) {
    const score = importanceScore(text)
    if (score < threshold) return { kept: false, reason: 'gate', score }
    const slot = extractSlot(text, maxSlotLen)
    if (slot !== null) {
      this.items = this.items.filter((it) => it.slot !== slot) // supersede
    }
    const item = { text, slot, score, ts: Date.now() }
    this.items.push(item)
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
    return scored.slice(0, topK)
  }

  get size() {
    return this.items.length
  }
}

// ---------------------------------------------------------------- dsh plugin
import z from 'schemastery'

export const Config = z.object({
  enabled: z.boolean().default(true),
  writeThreshold: z.number().default(0.6),
  slotMaxLen: z.number().default(6),
})

export const inject = [] // no required services; works standalone

export function apply(ctx, config) {
  const enabled = config.enabled ?? true
  const threshold = config.writeThreshold ?? 0.6
  const slotMaxLen = config.slotMaxLen ?? 6
  const store = new IgmStore()

  const log = (msg) => {
    // console.log is used deliberately: ctx.logger may not be injectable
    // without declaring it, and we must not risk another load failure.
    console.log(`[igm-memory] ${msg}`)
  }

  if (!enabled) {
    log('disabled by config')
    return
  }

  log(`enabled (threshold=${threshold}, slotMaxLen=${slotMaxLen})`)

  // Memory-write guard: expose a service so other plugins / the agent loop
  // can route memory candidates through the IGM gate.
  ctx.provide('igm.memory.write', (text) => {
    const res = store.add(text, threshold, slotMaxLen)
    log(res.kept ? `kept [${res.item.slot || 'none'}] ${text.slice(0, 40)}` : `filtered: ${text.slice(0, 40)}`)
    return res
  })

  ctx.provide('igm.memory.query', (text) => store.query(text, 3, slotMaxLen))

  ctx.provide('igm.memory.stats', () => ({
    stored: store.size,
    items: store.items.map((it) => ({ text: it.text, slot: it.slot })),
  }))

  log('services registered: igm.memory.write / igm.memory.query / igm.memory.stats')
}
