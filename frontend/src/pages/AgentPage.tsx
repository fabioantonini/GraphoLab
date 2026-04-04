import { useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Send, Paperclip, X, WifiOff, Loader2, ChevronDown, ChevronRight, Bot, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useAuthStore } from "@/store/auth"
import ReactMarkdown from "react-markdown"

interface Message {
  role: "user" | "assistant"
  text: string
  files?: string[]
}

interface Prompt {
  label: string
  text: string
}

export default function AgentPage() {
  const { t } = useTranslation()
  const [online, setOnline] = useState<boolean | null>(null)
  const [agentModel, setAgentModel] = useState<string>("")
  const [prompts, setPrompts] = useState<Prompt[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [streaming, setStreaming] = useState(false)
  const [attachedFiles, setAttachedFiles] = useState<File[]>([])
  const [expandedTools, setExpandedTools] = useState<Record<number, boolean>>({})
  const bottomRef = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Load status + prompts on mount
  useEffect(() => {
    const token = useAuthStore.getState().accessToken
    const headers = { Authorization: `Bearer ${token}` }

    fetch("/api/agent/status", { headers })
      .then((r) => r.json())
      .then((d) => {
        setOnline(d.ollama_reachable)
        setAgentModel(d.agent_model ?? "")
      })
      .catch(() => setOnline(false))

    fetch("/api/agent/prompts", { headers })
      .then((r) => r.json())
      .then((d) => setPrompts(d.prompts ?? []))
      .catch(() => {})
  }, [])

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = "auto"
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [input])

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = Array.from(e.target.files ?? [])
    setAttachedFiles((prev) => [...prev, ...picked])
    e.target.value = ""
  }

  function removeFile(idx: number) {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== idx))
  }

  function applyPrompt(text: string) {
    setInput(text)
    textareaRef.current?.focus()
  }

  function handleStop() {
    abortRef.current?.abort()
  }

  async function handleSend(e?: React.FormEvent) {
    e?.preventDefault()
    if (!input.trim() || streaming) return

    const userMsg = input.trim()
    const fileNames = attachedFiles.map((f) => f.name)
    setInput("")
    setAttachedFiles([])
    setMessages((m) => [...m, { role: "user", text: userMsg, files: fileNames }])
    setStreaming(true)

    const history = messages.map((m) => ({ role: m.role, content: m.text }))
    setMessages((m) => [...m, { role: "assistant", text: "" }])

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const token = useAuthStore.getState().accessToken
      const fd = new FormData()
      fd.append("message", userMsg)
      fd.append("history", JSON.stringify(history))
      for (const f of attachedFiles) fd.append("files", f)

      const res = await fetch("/api/agent/chat", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: fd,
        signal: controller.signal,
      })

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()
      while (reader) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value)
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data: ")) continue
          let text = ""
          try { text = JSON.parse(line.slice(6)) } catch { text = line.slice(6) }
          setMessages((m) => {
            const copy = [...m]
            copy[copy.length - 1] = { role: "assistant", text }
            return copy
          })
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") {
        // User stopped — keep whatever partial response was already shown
      } else {
        setMessages((m) => {
          const copy = [...m]
          copy[copy.length - 1] = { role: "assistant", text: `❌ ${t("common.error")} ${err}` }
          return copy
        })
      }
    } finally {
      abortRef.current = null
      setStreaming(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function toggleTools(idx: number) {
    setExpandedTools((prev) => ({ ...prev, [idx]: !prev[idx] }))
  }

  // Split assistant message into "main answer" and optional "<details>…</details>" section
  function splitMessage(text: string): { main: string; toolLog: string | null } {
    const detailsStart = text.indexOf("<details>")
    if (detailsStart === -1) return { main: text, toolLog: null }
    return {
      main: text.slice(0, detailsStart).trim(),
      toolLog: text.slice(detailsStart),
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-2rem)] max-w-4xl mx-auto p-4 gap-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bot className="h-5 w-5 text-primary" />
          <h1 className="text-xl font-semibold">{t("agent.title")}</h1>
        </div>
        <div className="flex items-center gap-2">
          {agentModel && (
            <Badge variant="secondary" className="text-xs font-mono">{agentModel}</Badge>
          )}
          {online === false && (
            <Badge variant="destructive" className="gap-1 text-xs">
              <WifiOff className="h-3 w-3" />
              {t("agent.offline")}
            </Badge>
          )}
        </div>
      </div>

      {/* Offline banner */}
      {online === false && (
        <div className="rounded-md border border-destructive/40 bg-destructive/5 px-4 py-2 text-sm text-destructive">
          {t("agent.offline_hint")}
        </div>
      )}

      {/* Suggested prompts */}
      {prompts.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {prompts.map((p) => (
            <button
              key={p.label}
              onClick={() => applyPrompt(p.text)}
              disabled={streaming || !online}
              className="rounded-full border bg-card px-3 py-1 text-xs font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-colors disabled:opacity-40"
            >
              {p.label}
            </button>
          ))}
        </div>
      )}

      {/* Chat area */}
      <div className="flex-1 overflow-y-auto rounded-lg border bg-muted/20 p-4 space-y-4 min-h-0">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground gap-2 py-12">
            <Bot className="h-10 w-10 opacity-30" />
            <p className="text-sm">{t("agent.empty_hint")}</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            {msg.role === "user" ? (
              <div className="max-w-[80%] space-y-1">
                <div className="rounded-lg bg-primary text-primary-foreground px-3 py-2 text-sm whitespace-pre-wrap">
                  {msg.text}
                </div>
                {msg.files && msg.files.length > 0 && (
                  <div className="flex flex-wrap gap-1 justify-end">
                    {msg.files.map((f) => (
                      <Badge key={f} variant="outline" className="text-xs gap-1">
                        <Paperclip className="h-3 w-3" />
                        {f}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="max-w-[85%] rounded-lg border bg-card px-3 py-2 text-sm space-y-2">
                {!msg.text ? (
                  <span className="animate-pulse text-muted-foreground">…</span>
                ) : (() => {
                  const { main, toolLog } = splitMessage(msg.text)
                  return (
                    <>
                      <div className="prose prose-sm max-w-none dark:prose-invert">
                        <ReactMarkdown>{main}</ReactMarkdown>
                      </div>
                      {toolLog && (
                        <div className="border-t pt-2 mt-2">
                          <button
                            onClick={() => toggleTools(i)}
                            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                          >
                            {expandedTools[i] ? (
                              <ChevronDown className="h-3 w-3" />
                            ) : (
                              <ChevronRight className="h-3 w-3" />
                            )}
                            {t("agent.tools_used")}
                          </button>
                          {expandedTools[i] && (
                            <div className="mt-1 text-xs text-muted-foreground prose prose-sm max-w-none dark:prose-invert">
                              <ReactMarkdown>{toolLog.replace(/<details>|<\/details>|<summary>[^<]*<\/summary>/g, "")}</ReactMarkdown>
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  )
                })()}
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <form onSubmit={handleSend} className="space-y-2">
        {/* Attached files */}
        {attachedFiles.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {attachedFiles.map((f, i) => (
              <Badge key={i} variant="secondary" className="gap-1 text-xs pr-1">
                <Paperclip className="h-3 w-3" />
                {f.name}
                <button
                  type="button"
                  onClick={() => removeFile(i)}
                  className="ml-0.5 rounded-full hover:bg-muted-foreground/20 p-0.5"
                >
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            ))}
          </div>
        )}

        <div className="flex gap-2 items-end">
          {/* File attachment */}
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="shrink-0 h-9 w-9"
            onClick={() => fileRef.current?.click()}
            disabled={streaming || !online}
            title={t("agent.attach_files")}
          >
            <Paperclip className="h-4 w-4" />
          </Button>
          <input
            ref={fileRef}
            type="file"
            multiple
            accept="image/*,.pdf"
            className="hidden"
            onChange={handleFileChange}
          />

          {/* Text input */}
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={online === false ? t("agent.offline") : t("agent.placeholder")}
            disabled={!online || streaming}
            rows={1}
            className="flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50 overflow-hidden"
          />

          {/* Send / Stop */}
          {streaming ? (
            <Button
              type="button"
              size="icon"
              variant="destructive"
              className="shrink-0 h-9 w-9"
              onClick={handleStop}
              title="Stop"
            >
              <Square className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              type="submit"
              size="icon"
              className="shrink-0 h-9 w-9"
              disabled={!online || !input.trim()}
            >
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>
        <p className="text-xs text-muted-foreground px-1">
          {t("agent.input_hint")}
        </p>
      </form>
    </div>
  )
}
