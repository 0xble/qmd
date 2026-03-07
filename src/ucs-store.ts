import { existsSync } from "node:fs"
import { openDatabase, type Database } from "./db.js"

const DEFAULT_RECENCY_WEIGHT = 0.25
const DEFAULT_RECENCY_HALF_LIFE_DAYS = 14

export type UcsTimelineRow = {
  itemId: string
  source: string
  kind: string
  uri: string
  title: string
  sourcePath: string
  sourceUri: string
  sourceOpenCmd: string
  revisionId: number
  revisionKey: string
  body: string
  eventAt: string
  ingestedAt: string
  lexicalRaw: number
  lexicalScore: number
  recencyScore: number
  totalScore: number
  matchedTerms: string[]
}

export type UcsEntityResult = {
  itemId: string
  source: string
  kind: string
  uri: string
  title: string
  latestEventAt: string
  sourcePath: string
  sourceUri: string
  sourceOpenCmd: string
  revisions: UcsRevision[]
  facts: UcsFact[]
}

export type UcsRevision = {
  id: number
  itemId: string
  revisionKey: string
  title: string
  body: string
  eventAt: string
  ingestedAt: string
  isCurrent: boolean
}

export type UcsFact = {
  id: string
  statement: string
  confidence: number
  freshness: string
  sourceCount: number
  createdAt: string
  updatedAt: string
}

export type UcsGetResult =
  | {
      type: "revision"
      uri: string
      revision: UcsRevision
      sourcePath: string
      sourceUri: string
      sourceOpenCmd: string
    }
  | {
      type: "fact"
      uri: string
      fact: UcsFact
    }

export type UcsDiffResult = {
  uri: string
  left: UcsRevision
  right: UcsRevision
  diff: string[]
}

type SearchOpts = {
  limit: number
  minScore: number
  dateRange?: {
    since?: string
    until?: string
  }
  collection?: string | string[]
}

function homeDir(): string {
  return process.env.HOME || "/tmp"
}

export function getDefaultUcsStorePath(): string {
  if (process.env.UCS_STORE_PATH) {
    return process.env.UCS_STORE_PATH
  }
  return `${homeDir()}/.local/share/context/_state/store.sqlite`
}

export function hasUcsStore(path: string = getDefaultUcsStorePath()): boolean {
  return existsSync(path)
}

export function openUcsStore(path: string = getDefaultUcsStorePath()): Database {
  return openDatabase(path)
}

function normalizeCollectionFilter(collection?: string | string[]): string[] {
  if (!collection) return []
  const values = Array.isArray(collection) ? collection : [collection]
  return values
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean)
}

function buildSourceFilterSql(collection?: string | string[]): {
  sql: string
  params: string[]
} {
  const filters = normalizeCollectionFilter(collection)
  if (filters.length === 0) {
    return { sql: "", params: [] }
  }
  const sourceFilters = filters
    .map((value) => {
      if (value === "sessions") return "session"
      if (value === "session") return "session"
      return value
    })
    .filter(Boolean)
  if (sourceFilters.length === 0) {
    return { sql: "", params: [] }
  }
  return {
    sql: ` AND i.source IN (${sourceFilters.map(() => "?").join(", ")})`,
    params: sourceFilters,
  }
}

function buildDateFilterSql(dateRange?: SearchOpts["dateRange"]): {
  sql: string
  params: string[]
} {
  if (!dateRange) return { sql: "", params: [] }
  const params: string[] = []
  const clauses: string[] = []
  if (dateRange.since) {
    clauses.push(`r.event_at >= ?`)
    params.push(dateRange.since)
  }
  if (dateRange.until) {
    clauses.push(`r.event_at <= ?`)
    params.push(dateRange.until)
  }
  if (clauses.length === 0) return { sql: "", params: [] }
  return {
    sql: ` AND ${clauses.join(" AND ")}`,
    params,
  }
}

function computeRecencyScore(eventAt: string): number {
  const eventMs = Date.parse(eventAt)
  if (Number.isNaN(eventMs)) return 0
  const nowMs = Date.now()
  const ageMs = Math.max(0, nowMs - eventMs)
  const ageDays = ageMs / (24 * 60 * 60 * 1000)
  return Math.exp((-Math.log(2) * ageDays) / DEFAULT_RECENCY_HALF_LIFE_DAYS)
}

function normalizeLexicalScores(values: number[]): number[] {
  const positives = values.map((value) => Math.max(0, value))
  const max = Math.max(...positives, 0)
  if (max <= 0) {
    return positives.map(() => 0)
  }
  return positives.map((value) => value / max)
}

function extractMatchedTerms(query: string, body: string, title: string): string[] {
  const haystack = `${title}\n${body}`.toLowerCase()
  return query
    .toLowerCase()
    .split(/\s+/)
    .map((term) => term.trim())
    .filter((term) => term.length >= 3)
    .filter((term, index, terms) => terms.indexOf(term) === index)
    .filter((term) => haystack.includes(term))
}

function makeSnippet(body: string, terms: string[]): string {
  const text = body.replace(/\s+/g, " ").trim()
  if (!text) return ""
  if (terms.length === 0) {
    return text.slice(0, 220)
  }
  const lower = text.toLowerCase()
  let offset = 0
  for (const term of terms) {
    const idx = lower.indexOf(term)
    if (idx >= 0) {
      offset = Math.max(0, idx - 80)
      break
    }
  }
  const snippet = text.slice(offset, offset + 220).trim()
  return offset > 0 ? `... ${snippet}` : snippet
}

export function searchUcsTimeline(db: Database, query: string, opts: SearchOpts): UcsTimelineRow[] {
  const trimmed = query.trim()
  if (!trimmed) return []
  const sourceFilter = buildSourceFilterSql(opts.collection)
  const dateFilter = buildDateFilterSql(opts.dateRange)
  const rawRows = db.prepare(
    `SELECT
      i.id AS item_id,
      i.source,
      i.kind,
      i.canonical_uri,
      i.title AS item_title,
      i.source_path,
      i.source_uri,
      i.source_open_cmd,
      r.id AS revision_id,
      r.revision_key,
      r.title AS revision_title,
      r.body,
      r.event_at,
      r.ingested_at,
      -bm25(corpus_fts, 4.0, 2.0, 1.0) AS lexical_raw
    FROM corpus_fts
    JOIN corpus_revision r ON r.id = corpus_fts.rowid
    JOIN corpus_item i ON i.id = r.item_id
    WHERE corpus_fts MATCH ?
      AND r.is_current = 1
      AND i.deleted_at = ''
      ${sourceFilter.sql}
      ${dateFilter.sql}
    LIMIT ?`
  ).all(trimmed, ...sourceFilter.params, ...dateFilter.params, Math.max(opts.limit * 4, 20)) as Array<Record<string, any>>
  if (rawRows.length === 0) return []

  const lexicalScores = normalizeLexicalScores(rawRows.map((row) => Number(row.lexical_raw ?? 0)))
  const rows = rawRows.map((row, index) => {
    const lexicalScore = lexicalScores[index] ?? 0
    const recencyScore = computeRecencyScore(String(row.event_at || row.ingested_at || ""))
    const totalScore = ((1 - DEFAULT_RECENCY_WEIGHT) * lexicalScore) + (DEFAULT_RECENCY_WEIGHT * recencyScore)
    const title = String(row.revision_title || row.item_title || row.canonical_uri || "")
    const body = String(row.body || "")
    return {
      itemId: String(row.item_id),
      source: String(row.source),
      kind: String(row.kind),
      uri: String(row.canonical_uri),
      title,
      sourcePath: String(row.source_path || ""),
      sourceUri: String(row.source_uri || ""),
      sourceOpenCmd: String(row.source_open_cmd || ""),
      revisionId: Number(row.revision_id),
      revisionKey: String(row.revision_key || ""),
      body,
      eventAt: String(row.event_at || ""),
      ingestedAt: String(row.ingested_at || ""),
      lexicalRaw: Number(row.lexical_raw || 0),
      lexicalScore,
      recencyScore,
      totalScore,
      matchedTerms: extractMatchedTerms(trimmed, body, title),
    } satisfies UcsTimelineRow
  })

  return rows
    .filter((row) => row.totalScore >= opts.minScore)
    .sort((a, b) => {
      const aEvent = Date.parse(a.eventAt || a.ingestedAt || "")
      const bEvent = Date.parse(b.eventAt || b.ingestedAt || "")
      if (!Number.isNaN(aEvent) && !Number.isNaN(bEvent) && aEvent !== bEvent) {
        return bEvent - aEvent
      }
      return b.totalScore - a.totalScore
    })
    .slice(0, opts.limit)
}

export function buildTimelineSnippet(row: UcsTimelineRow): string {
  return makeSnippet(row.body, row.matchedTerms)
}

export function listFacts(db: Database, query: string, limit: number): UcsFact[] {
  const trimmed = query.trim().toLowerCase()
  if (!trimmed) return []
  const terms = trimmed.split(/\s+/).filter(Boolean)
  const clauses = terms.map(() => `LOWER(statement) LIKE ?`).join(" AND ")
  const params = terms.map((term) => `%${term}%`)
  const rows = db.prepare(
    `SELECT id, statement, confidence, freshness, source_count, created_at, updated_at
     FROM fact
     WHERE ${clauses}
     ORDER BY confidence DESC, updated_at DESC
     LIMIT ?`
  ).all(...params, limit) as Array<Record<string, any>>
  return rows.map((row) => ({
    id: String(row.id),
    statement: String(row.statement || ""),
    confidence: Number(row.confidence || 0),
    freshness: String(row.freshness || ""),
    sourceCount: Number(row.source_count || 0),
    createdAt: String(row.created_at || ""),
    updatedAt: String(row.updated_at || ""),
  }))
}

export function getUcsDocument(db: Database, uri: string): UcsGetResult | null {
  const factMatch = uri.match(/^ucs:\/\/fact\/([^/]+)$/)
  if (factMatch?.[1]) {
    const fact = db.prepare(
      `SELECT id, statement, confidence, freshness, source_count, created_at, updated_at
       FROM fact WHERE id = ?`
    ).get(factMatch[1]) as UcsFact | null
    if (!fact) return null
    return {
      type: "fact",
      uri,
      fact,
    }
  }

  const parsed = parseUcsSessionUri(uri)
  if (!parsed) return null
  const revision = parsed.revisionId != null
    ? db.prepare(
        `SELECT r.id, r.item_id, r.revision_key, r.title, r.body, r.event_at, r.ingested_at, r.is_current
         FROM corpus_revision r
         WHERE r.id = ?`
      ).get(parsed.revisionId)
    : db.prepare(
        `SELECT r.id, r.item_id, r.revision_key, r.title, r.body, r.event_at, r.ingested_at, r.is_current
         FROM corpus_revision r
         JOIN corpus_item i ON i.current_revision_id = r.id
         WHERE i.id = ? AND i.deleted_at = ''`
      ).get(parsed.itemId)
  const item = db.prepare(
    `SELECT source_path, source_uri, source_open_cmd FROM corpus_item WHERE id = ?`
  ).get(parsed.itemId) as { source_path: string; source_uri: string; source_open_cmd: string } | null
  if (!revision || !item) return null
  return {
    type: "revision",
    uri,
    revision: {
      id: Number(revision.id),
      itemId: String(revision.item_id),
      revisionKey: String(revision.revision_key || ""),
      title: String(revision.title || ""),
      body: String(revision.body || ""),
      eventAt: String(revision.event_at || ""),
      ingestedAt: String(revision.ingested_at || ""),
      isCurrent: Number(revision.is_current) === 1,
    },
    sourcePath: String(item.source_path || ""),
    sourceUri: String(item.source_uri || ""),
    sourceOpenCmd: String(item.source_open_cmd || ""),
  }
}

export function parseUcsSessionUri(uri: string): {
  itemId: string
  revisionId?: number
} | null {
  const match = uri.match(/^ucs:\/\/session\/([^/]+)\/([^/]+)(?:\/revision\/(\d+))?$/)
  if (!match?.[1] || !match?.[2]) return null
  return {
    itemId: `session:${match[1]}:${match[2]}`,
    revisionId: match[3] ? Number(match[3]) : undefined,
  }
}

export function listEntity(db: Database, query: string): UcsEntityResult | null {
  const matches = searchUcsTimeline(db, query, {
    limit: 20,
    minScore: 0,
  })
  if (matches.length === 0) return null
  const [top] = matches
  if (!top) return null
  const revisions = db.prepare(
    `SELECT id, item_id, revision_key, title, body, event_at, ingested_at, is_current
     FROM corpus_revision
     WHERE item_id = ?
     ORDER BY event_at DESC, id DESC
     LIMIT 5`
  ).all(top.itemId) as Array<Record<string, any>>
  const facts = listFacts(db, query, 5)
  return {
    itemId: top.itemId,
    source: top.source,
    kind: top.kind,
    uri: top.uri,
    title: top.title,
    latestEventAt: top.eventAt || top.ingestedAt,
    sourcePath: top.sourcePath,
    sourceUri: top.sourceUri,
    sourceOpenCmd: top.sourceOpenCmd,
    revisions: revisions.map((revision) => ({
      id: Number(revision.id),
      itemId: String(revision.item_id),
      revisionKey: String(revision.revision_key || ""),
      title: String(revision.title || ""),
      body: String(revision.body || ""),
      eventAt: String(revision.event_at || ""),
      ingestedAt: String(revision.ingested_at || ""),
      isCurrent: Number(revision.is_current) === 1,
    })),
    facts,
  }
}

function getRevisionById(db: Database, id: number): UcsRevision | null {
  const revision = db.prepare(
    `SELECT id, item_id, revision_key, title, body, event_at, ingested_at, is_current
     FROM corpus_revision
     WHERE id = ?`
  ).get(id) as Record<string, any> | null
  if (!revision) return null
  return {
    id: Number(revision.id),
    itemId: String(revision.item_id),
    revisionKey: String(revision.revision_key || ""),
    title: String(revision.title || ""),
    body: String(revision.body || ""),
    eventAt: String(revision.event_at || ""),
    ingestedAt: String(revision.ingested_at || ""),
    isCurrent: Number(revision.is_current) === 1,
  }
}

function getCurrentAndPreviousRevision(db: Database, itemId: string, currentRevisionId?: number): {
  left: UcsRevision
  right: UcsRevision
} | null {
  const revisions = db.prepare(
    `SELECT id, item_id, revision_key, title, body, event_at, ingested_at, is_current
     FROM corpus_revision
     WHERE item_id = ?
     ORDER BY event_at DESC, id DESC`
  ).all(itemId)
  const normalized = (revisions as Array<Record<string, any>>).map((revision) => ({
    id: Number(revision.id),
    itemId: String(revision.item_id),
    revisionKey: String(revision.revision_key || ""),
    title: String(revision.title || ""),
    body: String(revision.body || ""),
    eventAt: String(revision.event_at || ""),
    ingestedAt: String(revision.ingested_at || ""),
    isCurrent: Number(revision.is_current) === 1,
  }))
  if (normalized.length < 2) return null
  if (currentRevisionId == null) {
    const right = normalized[0]
    const left = normalized[1]
    if (!left || !right) return null
    return {
      right,
      left,
    }
  }
  const index = normalized.findIndex((revision) => revision.id === currentRevisionId)
  if (index <= 0) return null
  const right = normalized[index]
  const left = normalized[index + 1] || normalized[index - 1]
  if (!left || !right) return null
  return {
    right,
    left,
  }
}

function diffLines(left: string, right: string): string[] {
  const a = left.split("\n")
  const b = right.split("\n")
  const m = a.length
  const n = b.length
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array.from({ length: n + 1 }, () => 0))
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      if (a[i] === b[j]) {
        dp[i]![j] = (dp[i + 1]![j + 1] || 0) + 1
      } else {
        dp[i]![j] = Math.max(dp[i + 1]![j] || 0, dp[i]![j + 1] || 0)
      }
    }
  }

  const out: string[] = []
  let i = 0
  let j = 0
  while (i < m && j < n) {
    if (a[i] === b[j]) {
      i++
      j++
      continue
    }
    if ((dp[i + 1]?.[j] || 0) >= (dp[i]?.[j + 1] || 0)) {
      out.push(`- ${a[i]}`)
      i++
      continue
    }
    out.push(`+ ${b[j]}`)
    j++
  }
  while (i < m) {
    out.push(`- ${a[i]}`)
    i++
  }
  while (j < n) {
    out.push(`+ ${b[j]}`)
    j++
  }
  return out
}

export function diffUcsDocument(
  db: Database,
  uri: string,
  opts: {
    revisionIds?: number[]
    previous?: boolean
  } = {}
): UcsDiffResult | null {
  const parsed = parseUcsSessionUri(uri)
  if (!parsed) return null
  let left: UcsRevision | null = null
  let right: UcsRevision | null = null

  if (opts.revisionIds && opts.revisionIds.length === 2) {
    left = getRevisionById(db, opts.revisionIds[0]!)
    right = getRevisionById(db, opts.revisionIds[1]!)
  } else {
    const pair = getCurrentAndPreviousRevision(db, parsed.itemId, parsed.revisionId)
    if (pair) {
      left = pair.left
      right = pair.right
    }
  }
  if (!left || !right) return null
  return {
    uri,
    left,
    right,
    diff: diffLines(left.body, right.body),
  }
}
