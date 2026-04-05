import { useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Upload, WifiOff, Loader2, FileText, CheckCircle2, Download } from "lucide-react"
import type { Components } from "react-markdown"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { complianceApi } from "@/lib/api"
import type { ComplianceBlock } from "@/lib/api"
import { useAuthStore } from "@/store/auth"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  parseReport,
  StructuredReport,
  type ParsedReport,
} from "@/components/ComplianceReport"

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
