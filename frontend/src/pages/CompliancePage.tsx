import { useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Upload, WifiOff, Loader2, FileText, CheckCircle2, Lightbulb, Download } from "lucide-react"
import type { Components } from "react-markdown"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { complianceApi, type ComplianceBlock } from "@/lib/api"
import { useAuthStore } from "@/store/auth"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

// ── Types ─────────────────────────────────────────────────────────────────────

interface ReqBlock {
  num: number
  name: string
  verdict: "✅" | "⚠️" | "❌" | null
  motivazione: string
  suggerimento: string | null
}

interface ParsedReport {
  blocks: ReqBlock[]
  conformi: number
  parziali: number
  mancanti: number
  judgment: string
}

// ── Parser ────────────────────────────────────────────────────────────────────

function parseReport(text: string): ParsedReport | null {
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

// ── Verdict config ────────────────────────────────────────────────────────────

const VC = {
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

function judgmentVariant(j: string): string {
  if (j.startsWith("Buona")) return "bg-green-100 text-green-800 border-green-300"
  if (j.startsWith("Conformità discreta")) return "bg-blue-100 text-blue-800 border-blue-300"
  if (j.startsWith("Conformità parziale")) return "bg-amber-100 text-amber-800 border-amber-300"
  return "bg-red-100 text-red-800 border-red-300"
}

// ── Markdown renderer (no @tailwindcss/typography needed) ────────────────────

const mdComponents: Components = {
  h1: ({ children }) => <h1 className="text-lg font-bold mt-4 mb-2">{children}</h1>,
  h2: ({ children }) => <h2 className="text-base font-semibold mt-4 mb-1.5 border-b pb-1">{children}</h2>,
  h3: ({ children }) => <h3 className="text-sm font-semibold mt-3 mb-1">{children}</h3>,
  p: ({ children }) => <p className="text-sm leading-relaxed mb-2">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  ul: ({ children }) => <ul className="list-disc list-inside text-sm space-y-0.5 mb-2 pl-2">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal list-inside text-sm space-y-0.5 mb-2 pl-2">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  hr: () => <hr className="my-4 border-border" />,
  code: ({ children }) => <code className="bg-muted px-1 py-0.5 rounded text-xs font-mono">{children}</code>,
  blockquote: ({ children }) => <blockquote className="border-l-2 border-muted-foreground/30 pl-3 italic text-muted-foreground text-sm">{children}</blockquote>,
  table: ({ children }) => <div className="overflow-x-auto mb-2"><table className="text-xs w-full border-collapse">{children}</table></div>,
  th: ({ children }) => <th className="border border-border px-2 py-1 bg-muted font-medium text-left">{children}</th>,
  td: ({ children }) => <td className="border border-border px-2 py-1">{children}</td>,
}

function MarkdownContent({ children }: { children: string }) {
  return (
    <div className="text-foreground">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>{children}</ReactMarkdown>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatCard({ emoji, count, total, label, className }: {
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

function ReqCard({ block }: { block: ReqBlock }) {
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

function StructuredReport({ report }: { report: ParsedReport }) {
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

// ── Main page ─────────────────────────────────────────────────────────────────

export default function CompliancePage() {
  const { t } = useTranslation()
  const [online, setOnline] = useState<boolean | null>(null)
  const [filename, setFilename] = useState<string | null>(null)
  const [streaming, setStreaming] = useState(false)
  const [result, setResult] = useState<string>("")
  const [done, setDone] = useState(false)
  const [pdfLoading, setPdfLoading] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    complianceApi.status().then(({ data }) => setOnline(data.ollama_reachable))
  }, [])

  useEffect(() => {
    if (streaming) bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [result, streaming])

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ""

    setFilename(file.name)
    setResult("")
    setDone(false)
    setStreaming(true)

    try {
      const token = useAuthStore.getState().accessToken
      const fd = new FormData()
      fd.append("file", file)

      const res = await fetch("/api/compliance/check", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: fd,
      })

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()

      while (reader) {
        const { done: readerDone, value } = await reader.read()
        if (readerDone) break
        const chunk = decoder.decode(value)
        for (const line of chunk.split("\n")) {
          if (line.startsWith("data: ")) {
            try { setResult(JSON.parse(line.slice(6))) }
            catch { setResult(line.slice(6)) }
          }
        }
      }
      setDone(true)
    } finally {
      setStreaming(false)
    }
  }

  const parsed = done ? parseReport(result) : null

  async function handleDownloadPdf() {
    if (!parsed || !filename) return
    setPdfLoading(true)
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
      setPdfLoading(false)
    }
  }

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold">{t("compliance.title")}</h1>
        <p className="text-sm text-muted-foreground mt-1">{t("compliance.subtitle")}</p>
      </div>

      {online === false && (
        <div className="flex items-center gap-2 rounded-md border border-destructive/40 bg-destructive/5 px-4 py-2 text-sm text-destructive">
          <WifiOff className="h-4 w-4 shrink-0" />
          {t("rag.offline")}
        </div>
      )}

      {/* Upload card */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">{t("compliance.upload_title")}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <Button
              variant="outline"
              className="gap-2"
              onClick={() => fileRef.current?.click()}
              disabled={!online || streaming}
            >
              {streaming
                ? <Loader2 className="h-4 w-4 animate-spin" />
                : <Upload className="h-4 w-4" />}
              {streaming ? t("compliance.analyzing") : t("compliance.upload_btn")}
            </Button>
            <input ref={fileRef} type="file" accept=".pdf" className="hidden" onChange={handleFileChange} />
            {filename && (
              <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
                <FileText className="h-4 w-4 shrink-0" />
                <span className="truncate max-w-[280px]">{filename}</span>
                {done && <CheckCircle2 className="h-4 w-4 text-green-600 shrink-0" />}
              </div>
            )}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">{t("compliance.upload_hint")}</p>
        </CardContent>
      </Card>

      {/* Result */}
      {result && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center justify-between">
              {t("compliance.result_title")}
              <div className="flex items-center gap-2">
                {streaming && (
                  <span className="text-xs text-muted-foreground font-normal flex items-center gap-1.5">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    {t("compliance.analyzing")}
                  </span>
                )}
                {done && parsed && (
                  <Button size="sm" variant="outline" className="gap-1.5 h-7 text-xs" onClick={handleDownloadPdf} disabled={pdfLoading}>
                    {pdfLoading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Download className="h-3 w-3" />}
                    {t("compliance.download_pdf")}
                  </Button>
                )}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {/* Streaming view: markdown rendered in scrollable area */}
            {streaming && (
              <div className="max-h-[600px] overflow-y-auto">
                <MarkdownContent>{result}</MarkdownContent>
                <span className="inline-block animate-pulse text-muted-foreground">▊</span>
                <div ref={bottomRef} />
              </div>
            )}

            {/* Done view: structured cards */}
            {done && parsed && <StructuredReport report={parsed} />}

            {/* Done but parsing failed: fallback to markdown */}
            {done && !parsed && <MarkdownContent>{result}</MarkdownContent>}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
