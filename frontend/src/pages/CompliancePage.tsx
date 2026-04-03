import { useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Upload, WifiOff, Loader2, FileText, CheckCircle2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { complianceApi } from "@/lib/api"
import { useAuthStore } from "@/store/auth"

export default function CompliancePage() {
  const { t } = useTranslation()
  const [online, setOnline] = useState<boolean | null>(null)
  const [filename, setFilename] = useState<string | null>(null)
  const [streaming, setStreaming] = useState(false)
  const [result, setResult] = useState<string>("")
  const [done, setDone] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    complianceApi.status().then(({ data }) => setOnline(data.ollama_reachable))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [result])

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
            try {
              const text = JSON.parse(line.slice(6))
              setResult(text)
            } catch {
              setResult(line.slice(6))
            }
          }
        }
      }
      setDone(true)
    } finally {
      setStreaming(false)
    }
  }

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
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
            <input
              ref={fileRef}
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={handleFileChange}
            />
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
            <CardTitle className="text-sm">{t("compliance.result_title")}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="rounded-md bg-muted/30 border p-4 text-sm whitespace-pre-wrap font-mono leading-relaxed max-h-[600px] overflow-y-auto">
              {result}
              {streaming && <span className="animate-pulse text-muted-foreground"> ▊</span>}
              <div ref={bottomRef} />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
