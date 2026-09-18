// Reproducible storage-layer microbenchmark for the DSH IGM plugin.
//
// It measures JSON persistence, slot updates, current reads, and historical
// reads only. It deliberately does not measure embeddings, an LLM, DSH boot,
// or a task's answer quality.

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const { IgmStore } = await import(
  pathToFileURL(path.join(root, 'lib/index.js')).href,
)

const baseFacts = Number.parseInt(process.env.IGM_BENCH_FACTS || '300', 10)
const updates = Number.parseInt(process.env.IGM_BENCH_UPDATES || '100', 10)
const reads = Number.parseInt(process.env.IGM_BENCH_READS || '100', 10)
if (![baseFacts, updates, reads].every((value) => Number.isInteger(value) && value > 0)) {
  throw new Error('IGM_BENCH_FACTS, IGM_BENCH_UPDATES, and IGM_BENCH_READS must be positive integers')
}

const nowMs = () => Number(process.hrtime.bigint()) / 1e6
const quantile = (values, q) => {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * q) - 1))]
}
const timing = (values) => ({
  count: values.length,
  medianMs: Number(quantile(values, 0.5).toFixed(3)),
  p95Ms: Number(quantile(values, 0.95).toFixed(3)),
  totalMs: Number(values.reduce((sum, value) => sum + value, 0).toFixed(3)),
})
const fact = (index, value = 0) => `我的偏好${String(index).padStart(3, '0')}是值${value}。`

function measure(operation, samples, action) {
  for (let index = 0; index < samples; index++) {
    const started = nowMs()
    action(index)
    operation.push(nowMs() - started)
  }
}

function runCase(name, defaults = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `igm-bench-${name}-`))
  const file = path.join(dir, 'memory.json')
  try {
    const store = new IgmStore(file, defaults)
    const writes = []
    measure(writes, baseFacts, (index) => store.add(fact(index)))
    const slotUpdates = []
    measure(slotUpdates, updates, (index) => store.add(fact(index % baseFacts, index + 1)))
    const queries = []
    measure(queries, reads, (index) => store.query(`我的偏好${String(index % baseFacts).padStart(3, '0')}是什么？`))
    const history = []
    if (typeof store.history === 'function') {
      measure(history, reads, (index) => store.history(`偏好${String(index % baseFacts).padStart(3, '0')}`))
    }
    const diskBytes = fs.statSync(file).size
    if (global.gc) global.gc()
    return {
      mode: name,
      activeItems: store.size,
      eventCount: store.eventCount ?? store.size,
      diskBytes,
      write: timing(writes),
      update: timing(slotUpdates),
      query: timing(queries),
      history: history.length ? timing(history) : null,
      heapUsedBytesAfterGc: process.memoryUsage().heapUsed,
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
}

const result = {
  kind: 'dsh_igm_memory_storage_microbenchmark',
  scope: 'local JSON persistence and in-process store operations only',
  config: { baseFacts, updates, reads, node: process.version, platform: process.platform, arch: process.arch },
  results: [
    runCase('delete', { supersede: 'delete' }),
    runCase('archive', { supersede: 'archive' }),
  ],
}
console.log(JSON.stringify(result, null, 2))
