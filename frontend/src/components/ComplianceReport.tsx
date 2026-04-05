/**
 * Shared compliance report components and utilities.
 * Used by CompliancePage and inline in AgentPage / AgentProjectPage.
 */

import { useState } from "react"
import { Lightbulb, Download, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { complianceApi, type ComplianceBlock } from "@/lib/api"

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ReqBlock {
  num: number
  name: string
  verdict: "✅" | "⚠️" | "❌" | null
  motivazione: string
  suggerimento: string | null
}

export interface ParsedReport {
  blocks: ReqBlock[]
  conformi: number
  parziali: number
  mancanti: number
  judgment: string
}

// ── Parser ────────────────────────────────────────────────────────────────────

export function parseReport(text: string): ParsedReport | null {
  const blocks: ReqBlock[] = []
  const reqMatches = [...text.matchAll(/REQ-(\d{1,2})\.\s*([^\n]+)/g)]

  for (let i = 0; i < reqMatches.length; i++) {
    const match = reqMatches[i]
    const num = parseInt(match[1])
    if (num < 1 || num > 20) continue

    const name = match[2].trim().replace(/\*+/g, "").trimEnd()
    const start = match.index!
    const nextMatch = reqMatches.find((m, idx) => idx > i && parseInt(m[1]) > num)
    const end = nextMatch ? nextMatch.index! : text.indexOf("\n---")
    const blockText = text.slice(start, end > start ? end : undefined)

    const verdictM = blockText.match(/Verdetto:\s*(✅|⚠️|❌)/)
    const verdict = (verdictM?.[1] as "✅" | "⚠️" | "❌") ?? null

    const motivM = blockText.match(/Motivazione:\s*([\s\S]+?)(?=\s*💡\s*Suggerimento|\n\s*\n\s*\n|$)/)
    const motivazione = motivM?.[1]?.trim() ?? ""

    const suggM = blockText.match(/💡\s*Suggerimento:\s*([\s\S]+?)$/)
    const rawSugg = suggM?.[1]?.trim() ?? null
    const suggerimento = rawSugg && !/^nessuno\.?$/i.test(rawSugg) ? rawSugg : null

    blocks.push({ num, name, verdict, motivazione, suggerimento })
  }

  if (blocks.length < 10) return null

  const confM = text.match(/✅ Conformi:\s*(\d+)/)
  const parzM = text.match(/⚠️ Parziali:\s*(\d+)/)
  const mancM = text.match(/❌ Mancanti:\s*(\d+)/)
  const judgM = text.match(/\*\*Giudizio complessivo:\s*([^*]+)\*\*/)

  return {
    blocks,
    conformi: confM ? parseInt(confM[1]) : blocks.filter(b => b.verdict === "✅").length,
    parziali: parzM ? parseInt(parzM[1]) : blocks.filter(b => b.verdict === "⚠️").length,
    mancanti: mancM ? parseInt(mancM[1]) : blocks.filter(b => b.verdict === "❌").length,
    judgment: judgM?.[1]?.trim() ?? "",
  }
}

// ── Compliance marker parser (for agent chat output) ─────────────────────────

export function parseComplianceMarker(text: string): {
  mainText: string
  complianceData: (ParsedReport & { filename: string }) | null
} {
  const m = text.match(/<!-- COMPLIANCE_REPORT: (\{[\s\S]+?\}) -->/)
  if (!m) return { mainText: text, complianceData: null }
  try {
    return {
      mainText: text.slice(0, text.indexOf("<!-- COMPLIANCE_REPORT:")).trim(),
      complianceData: JSON.parse(m[1]),
    }
  } catch {
    return { mainText: text, complianceData: null }
  }
}

// ── Verdict config ────────────────────────────────────────────────────────────

export const VC = {
  "✅": {
    label: "CONFORME",
    border: "border-l-green-500",
    badge: "bg-green-100 text-green-800 border-green-300",
    sugg: "bg-green-50 text-green-700",
  },
  "⚠️": {
    label: "PARZIALE",
    border: "border-l-amber-500",
    badge: "bg-amber-100 text-amber-800 border-amber-300",
    sugg: "bg-amber-50 text-amber-800",
  },
  "❌": {
    label: "MANCANTE",
    border: "border-l-red-500",
    badge: "bg-red-100 text-red-800 border-red-300",
    sugg: "bg-red-50 text-red-800",
  },
} as const

export function judgmentVariant(j: string): string {
  if (j.startsWith("Buona")) return "bg-green-100 text-green-800 border-green-300"
  if (j.startsWith("Conformità discreta")) return "bg-blue-100 text-blue-800 border-blue-300"
  if (j.startsWith("Conformità parziale")) return "bg-amber-100 text-amber-800 border-amber-300"
  return "bg-red-100 text-red-800 border-red-300"
}

// ── Sub-components ────────────────────────────────────────────────────────────

export function StatCard({ emoji, count, total, label, className }: {
  emoji: string; count: number; total: number; label: string; className: string
}) {
  return (
    <div className={`rounded-xl border px-5 py-4 flex flex-col gap-1 ${className}`}>
      <div className="flex items-baseline gap-1.5">
        <span className="text-3xl font-bold">{count}</span>
        <span className="text-sm text-muted-foreground font-medium">/ {total}</span>
      </div>
      <div className="text-sm font-medium">{emoji} {label}</div>
    </div>
  )
}

export function ReqCard({ block }: { block: ReqBlock }) {
  const cfg = block.verdict ? VC[block.verdict] : null
  return (
    <div className={`border border-l-4 rounded-lg ${cfg?.border ?? "border-l-muted"} bg-card shadow-sm overflow-hidden`}>
      <div className="px-4 pt-3 pb-2 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <span className="text-[10px] font-mono text-muted-foreground tracking-wider">
            REQ-{String(block.num).padStart(2, "0")}
          </span>
          <p className="font-semibold text-sm leading-snug mt-0.5">{block.name}</p>
        </div>
        {cfg && (
          <span className={`shrink-0 text-[11px] font-semibold px-2 py-0.5 rounded-full border whitespace-nowrap ${cfg.badge}`}>
            {block.verdict} {cfg.label}
          </span>
        )}
      </div>
      <div className="px-4 pb-3 space-y-2">
        <p className="text-sm text-muted-foreground leading-relaxed">{block.motivazione}</p>
        {block.suggerimento && (
          <div className={`flex gap-2 rounded-md px-3 py-2 text-xs leading-relaxed ${cfg?.sugg ?? "bg-muted"}`}>
            <Lightbulb className="h-3.5 w-3.5 mt-0.5 shrink-0" />
            <span>{block.suggerimento}</span>
          </div>
        )}
      </div>
    </div>
  )
}

type Filter = "all" | "✅" | "⚠️" | "❌"

export function StructuredReport({ report }: { report: ParsedReport }) {
  const [filter, setFilter] = useState<Filter>("all")
  const total = report.blocks.length

  const visible = filter === "all"
    ? report.blocks
    : report.blocks.filter(b => b.verdict === filter)

  const tabs: { key: Filter; label: string; count: number }[] = [
    { key: "all", label: "Tutti", count: total },
    { key: "✅", label: "Conformi", count: report.conformi },
    { key: "⚠️", label: "Parziali", count: report.parziali },
    { key: "❌", label: "Mancanti", count: report.mancanti },
  ]

  return (
    <div className="space-y-6">
      {/* Summary stats */}
      <div className="grid grid-cols-3 gap-3">
        <StatCard emoji="✅" count={report.conformi} total={total} label="Conformi"
          className="border-green-200 bg-green-50/60 text-green-900" />
        <StatCard emoji="⚠️" count={report.parziali} total={total} label="Parziali"
          className="border-amber-200 bg-amber-50/60 text-amber-900" />
        <StatCard emoji="❌" count={report.mancanti} total={total} label="Mancanti"
          className="border-red-200 bg-red-50/60 text-red-900" />
      </div>

      {/* Judgment */}
      {report.judgment && (
        <div className={`rounded-xl border px-4 py-3 text-sm font-medium ${judgmentVariant(report.judgment)}`}>
          {report.judgment}
        </div>
      )}

      {/* Filter tabs */}
      <div className="flex gap-1.5 flex-wrap">
        {tabs.map(tab => (
          <button
            key={tab.key}
            onClick={() => setFilter(tab.key)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
              filter === tab.key
                ? "bg-primary text-primary-foreground border-primary"
                : "bg-background text-muted-foreground border-border hover:border-foreground/30"
            }`}
          >
            {tab.label} <span className="opacity-70">({tab.count})</span>
          </button>
        ))}
      </div>

      {/* REQ cards */}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {visible.map(block => (
          <ReqCard key={block.num} block={block} />
        ))}
      </div>
    </div>
  )
}

// ── Download PDF button ───────────────────────────────────────────────────────

export function ComplianceDownloadButton({
  parsed,
  filename,
}: {
  parsed: ParsedReport
  filename: string
}) {
  const [loading, setLoading] = useState(false)

  async function handleDownload() {
    setLoading(true)
    try {
      const res = await complianceApi.pdf({
        filename,
        blocks: parsed.blocks as ComplianceBlock[],
        conformi: parsed.conformi,
        parziali: parsed.parziali,
        mancanti: parsed.mancanti,
        judgment: parsed.judgment,
      })
      const url = URL.createObjectURL(new Blob([res.data], { type: "application/pdf" }))
      const a = document.createElement("a")
      a.href = url
      a.download = `compliance_${filename.replace(/\.pdf$/i, "")}.pdf`
      a.click()
      URL.revokeObjectURL(url)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Button size="sm" variant="outline" className="gap-1.5 h-7 text-xs" onClick={handleDownload} disabled={loading}>
      {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Download className="h-3 w-3" />}
      Scarica PDF
    </Button>
  )
}
