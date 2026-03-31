import { useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Send, Upload, Trash2, WifiOff } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ragApi } from "@/lib/api"
import { useAuthStore } from "@/store/auth"

interface Message { role: "user" | "assistant"; text: string }

export default function RagPage() {
  const { t } = useTranslation()
  const { accessToken } = useAuthStore()
  const [online, setOnline] = useState<boolean | null>(null)
  const [docs, setDocs] = useState<{ filename: string; chunks: number }[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [streaming, setStreaming] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    ragApi.status().then(({ data }) => setOnline(data.ollama_reachable))
    ragApi.listDocs().then(({ data }) => setDocs(data))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  async function handleSend(e: React.FormEvent) {
    e.preventDefault()
    if (!input.trim() || streaming) return

    const userMsg = input.trim()
    setInput("")
    setMessages((m) => [...m, { role: "user", text: userMsg }])
    setStreaming(true)

    // Build history in [[user, assistant], ...] format
    const history: string[][] = []
    const msgs = [...messages]
    for (let i = 0; i < msgs.length - 1; i += 2) {
      if (msgs[i]?.role === "user" && msgs[i + 1]?.role === "assistant") {
        history.push([msgs[i].text, msgs[i + 1].text])
      }
    }

    // SSE stream
    let assistantText = ""
    setMessages((m) => [...m, { role: "assistant", text: "" }])

    try {
      const res = await fetch("/api/rag/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ message: userMsg, history }),
      })
      const reader = res.body?.getReader()
      const decoder = new TextDecoder()
      while (reader) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value)
        for (const line of chunk.split("\n")) {
          if (line.startsWith("data: ")) {
            assistantText += line.slice(6)
            setMessages((m) => {
              const copy = [...m]
              copy[copy.length - 1] = { role: "assistant", text: assistantText }
              return copy
            })
          }
        }
      }
    } finally {
      setStreaming(false)
    }
  }

  async function handleUploadDoc(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    await ragApi.addDoc(file)
    const { data } = await ragApi.listDocs()
    setDocs(data)
    e.target.value = ""
  }

  async function handleRemoveDoc(filename: string) {
    await ragApi.removeDoc(filename)
    setDocs((d) => d.filter((x) => x.filename !== filename))
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <h1 className="text-2xl font-semibold">{t("rag.title")}</h1>

      {online === false && (
        <div className="flex items-center gap-2 rounded-md border border-destructive/40 bg-destructive/5 px-4 py-2 text-sm text-destructive">
          <WifiOff className="h-4 w-4" />
          {t("rag.offline")}
        </div>
      )}

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Chat */}
        <div className="lg:col-span-2 flex flex-col gap-3">
          <div className="flex-1 rounded-lg border bg-muted/20 p-4 min-h-[400px] max-h-[500px] overflow-y-auto space-y-3">
            {messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[80%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
                    m.role === "user" ? "bg-primary text-primary-foreground" : "bg-card border"
                  }`}
                >
                  {m.text || <span className="animate-pulse text-muted-foreground">…</span>}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
          <form onSubmit={handleSend} className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={t("rag.placeholder")}
              disabled={!online || streaming}
            />
            <Button type="submit" disabled={!online || streaming || !input.trim()}>
              <Send className="h-4 w-4" />
            </Button>
          </form>
        </div>

        {/* Knowledge base docs */}
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm">{t("rag.docs_title")}</CardTitle>
              <Button variant="outline" size="sm" className="h-7 text-xs gap-1" onClick={() => fileRef.current?.click()}>
                <Upload className="h-3.5 w-3.5" />
                {t("rag.add_doc")}
              </Button>
              <input ref={fileRef} type="file" accept=".pdf,.docx,.txt" className="hidden" onChange={handleUploadDoc} />
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            {docs.length === 0 ? (
              <p className="text-xs text-muted-foreground">{t("rag.no_docs")}</p>
            ) : (
              docs.map((doc) => (
                <div key={doc.filename} className="flex items-center gap-2 text-sm">
                  <span className="truncate flex-1 text-xs">{doc.filename}</span>
                  <Badge variant="secondary" className="text-xs">{doc.chunks}</Badge>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-muted-foreground hover:text-destructive shrink-0"
                    onClick={() => handleRemoveDoc(doc.filename)}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
