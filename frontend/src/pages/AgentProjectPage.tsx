import { useEffect, useRef, useState } from "react"
import { useParams, useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"
import {
  Send, Paperclip, X, WifiOff, ChevronDown, ChevronRight, Bot, Square,
  Plus, Trash2, Check, MessageSquare, FolderOpen, FileText, AlertTriangle,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  DropdownMenu, DropdownMenuTrigger, DropdownMenuContent,
  DropdownMenuItem, DropdownMenuSeparator, DropdownMenuLabel,
} from "@/components/ui/dropdown-menu"
import { cn } from "@/lib/utils"
import { useAuthStore } from "@/store/auth"
import ReactMarkdown from "react-markdown"
import { api, agentProjectsApi, type AgentChat, type AgentMessage, type Document } from "@/lib/api"

// ── AuthImage ────────────────────────────────────────────────────────────────

function AuthImage({ src, alt }: { src: string; alt: string }) {
  const [blobSrc, setBlobSrc] = useState<string | null>(null)
  useEffect(() => {
    const path = src.startsWith("/api/") ? src.slice(4) : src
    api.get<Blob>(path, { responseType: "blob" })
      .then(({ data }) => setBlobSrc(URL.createObjectURL(data)))
      .catch(() => setBlobSrc(null))
    return () => { if (blobSrc) URL.revokeObjectURL(blobSrc) }
  }, [src])
  if (!blobSrc) return null
  return <img src={blobSrc} alt={alt} className="max-w-full rounded border my-2 max-h-64 object-contain" />
}

// ── Types ─────────────────────────────────────────────────────────────────────

interface UiMessage {
  role: "user" | "assistant"
  text: string
  files?: string[]
}

interface Prompt {
  label: string
  text: string
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function splitMessage(text: string): { main: string; toolLog: string | null } {
  const idx = text.indexOf("<details>")
  if (idx === -1) return { main: text, toolLog: null }
  return { main: text.slice(0, idx).trim(), toolLog: text.slice(idx) }
}

function formatBytes(b: number) {
  if (b < 1024) return `${b} B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1024 / 1024).toFixed(1)} MB`
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function AgentProjectPage() {
  const { projectId } = useParams<{ projectId: string }>()
  const navigate = useNavigate()
  const { t } = useTranslation()
  const pid = Number(projectId)

  // Project / chat state
  const [projectTitle, setProjectTitle] = useState("")
  const [chats, setChats] = useState<AgentChat[]>([])
  const [activeChatId, setActiveChatId] = useState<number | null>(null)
  const [documents, setDocuments] = useState<Document[]>([])
  const [confirmDeleteProject, setConfirmDeleteProject] = useState(false)
  const [confirmDeleteChat, setConfirmDeleteChat] = useState<number | null>(null)

  // Chat UI state
  const [online, setOnline] = useState<boolean | null>(null)
  const [agentModel, setAgentModel] = useState("")
  const [prompts, setPrompts] = useState<Prompt[]>([])
  const [messages, setMessages] = useState<UiMessage[]>([])
  const [input, setInput] = useState("")
  const [streaming, setStreaming] = useState(false)
  const [attachedFiles, setAttachedFiles] = useState<File[]>([])
  const [reusedDocs, setReusedDocs] = useState<Document[]>([])
  const [selectedPrompts, setSelectedPrompts] = useState<Prompt[]>([])
  const [expandedTools, setExpandedTools] = useState<Record<number, boolean>>({})

  const bottomRef = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  // ── Load project data ────────────────────────────────────────────────────

  useEffect(() => {
    loadProject()
    loadStatus()
  }, [pid])

  async function loadProject() {
    try {
      const [chatsRes, docsRes] = await Promise.all([
        agentProjectsApi.listChats(pid),
        agentProjectsApi.listDocuments(pid),
      ])
      setChats(chatsRes.data)
      setDocuments(docsRes.data)
      // Auto-select first chat
      if (chatsRes.data.length > 0 && activeChatId === null) {
        await selectChat(chatsRes.data[0].id)
      }
    } catch {
      // project may not exist
    }
  }

  async function loadStatus() {
    const token = useAuthStore.getState().accessToken
    const headers = { Authorization: `Bearer ${token}` }
    try {
      const [statusRes, promptsRes] = await Promise.all([
        fetch("/api/agent/status", { headers }).then(r => r.json()),
        fetch("/api/agent/prompts", { headers }).then(r => r.json()),
      ])
      setOnline(statusRes.ollama_reachable)
      setAgentModel(statusRes.agent_model ?? "")
      setPrompts(promptsRes.prompts ?? [])
      // Try to get project title from projects list
      const projRes = await agentProjectsApi.listProjects()
      const proj = projRes.data.find(p => p.id === pid)
      if (proj) setProjectTitle(proj.title)
    } catch {
      setOnline(false)
    }
  }

  async function selectChat(chatId: number) {
    setActiveChatId(chatId)
    setMessages([])
    setInput("")
    setAttachedFiles([])
    setReusedDocs([])
    setSelectedPrompts([])
    try {
      const { data } = await agentProjectsApi.getChat(chatId)
      const uiMsgs: UiMessage[] = data.messages.map((m: AgentMessage) => ({
        role: m.role as "user" | "assistant",
        text: m.content,
        files: m.file_ids.length > 0 ? m.file_ids.map(String) : undefined,
      }))
      setMessages(uiMsgs)
    } catch {
      // empty chat
    }
  }

  async function handleNewChat() {
    try {
      const { data } = await agentProjectsApi.createChat(pid)
      setChats(prev => [data, ...prev])
      await selectChat(data.id)
    } catch {}
  }

  async function handleDeleteChat(chatId: number) {
    try {
      await agentProjectsApi.deleteChat(chatId)
      setChats(prev => prev.filter(c => c.id !== chatId))
      if (activeChatId === chatId) {
        const remaining = chats.filter(c => c.id !== chatId)
        if (remaining.length > 0) await selectChat(remaining[0].id)
        else { setActiveChatId(null); setMessages([]) }
      }
    } catch {}
    setConfirmDeleteChat(null)
  }

  async function handleDeleteProject() {
    try {
      await agentProjectsApi.deleteProject(pid)
      navigate("/agent")
    } catch {}
    setConfirmDeleteProject(false)
  }

  async function handleDeleteDocument(docId: number) {
    try {
      await agentProjectsApi.deleteDocument(pid, docId)
      setDocuments(prev => prev.filter(d => d.id !== docId))
      setReusedDocs(prev => prev.filter(d => d.id !== docId))
    } catch {}
  }

  // ── Chat handlers ────────────────────────────────────────────────────────

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = "auto"
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [input])

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = Array.from(e.target.files ?? [])
    setAttachedFiles(prev => [...prev, ...picked])
    e.target.value = ""
  }

  function removeFile(idx: number) {
    setAttachedFiles(prev => prev.filter((_, i) => i !== idx))
  }

  function removeReusedDoc(docId: number) {
    setReusedDocs(prev => prev.filter(d => d.id !== docId))
  }

  function toggleReusedDoc(doc: Document) {
    setReusedDocs(prev =>
      prev.some(d => d.id === doc.id)
        ? prev.filter(d => d.id !== doc.id)
        : [...prev, doc]
    )
  }

  function togglePrompt(p: Prompt) {
    setSelectedPrompts(prev =>
      prev.some(x => x.label === p.label)
        ? prev.filter(x => x.label !== p.label)
        : [...prev, p]
    )
  }

  function applyPrompt(text: string) {
    setInput(text)
    textareaRef.current?.focus()
  }

  function toggleTools(idx: number) {
    setExpandedTools(prev => ({ ...prev, [idx]: !prev[idx] }))
  }

  async function handleSend(e?: React.FormEvent) {
    e?.preventDefault()
    if ((!input.trim() && selectedPrompts.length === 0) || streaming) return

    // Create chat on-demand if none active
    let chatId = activeChatId
    if (chatId === null) {
      const { data } = await agentProjectsApi.createChat(pid)
      setChats(prev => [data, ...prev])
      setActiveChatId(data.id)
      chatId = data.id
    }

    const base = selectedPrompts.map(p => p.text).join("\n\n")
    const extra = input.trim()
    const fullMsg = base && extra ? `${base}\n\n${extra}` : base || extra
    const userLabel = selectedPrompts.length > 0
      ? selectedPrompts.map(p => p.label).join(" · ") + (extra ? ` — ${extra}` : "")
      : extra
    const fileNames = [
      ...attachedFiles.map(f => f.name),
      ...reusedDocs.map(d => d.filename),
    ]

    setInput("")
    setAttachedFiles([])
    setReusedDocs([])
    setSelectedPrompts([])
    setMessages(m => [...m, { role: "user", text: userLabel, files: fileNames }])
    setStreaming(true)

    const history = messages.map(m => ({ role: m.role, content: m.text }))
    setMessages(m => [...m, { role: "assistant", text: "" }])

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const token = useAuthStore.getState().accessToken
      const fd = new FormData()
      fd.append("message", fullMsg)
      fd.append("history", JSON.stringify(history))
      fd.append("project_id", String(pid))
      fd.append("chat_id", String(chatId))
      fd.append("reused_doc_ids", JSON.stringify(reusedDocs.map(d => d.id)))
      for (const f of attachedFiles) fd.append("files", f)

      const res = await fetch("/api/agent/chat", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: fd,
        signal: controller.signal,
      })

      // Refresh documents immediately after the request is accepted — the backend
      // saves uploaded files before starting to stream, so they are already in DB.
      agentProjectsApi.listDocuments(pid).then(r => setDocuments(r.data)).catch(() => {})

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()
      let lineBuf = ""
      while (reader) {
        const { done, value } = await reader.read()
        if (done) break
        lineBuf += decoder.decode(value, { stream: true })
        const lines = lineBuf.split("\n")
        lineBuf = lines.pop() ?? ""
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          let text = ""
          try { text = JSON.parse(line.slice(6)) } catch { text = line.slice(6) }
          setMessages(m => {
            const copy = [...m]
            copy[copy.length - 1] = { role: "assistant", text }
            return copy
          })
        }
      }
      // Refresh docs and chat list after send (new uploads may have appeared)
      agentProjectsApi.listDocuments(pid).then(r => setDocuments(r.data)).catch(() => {})
      agentProjectsApi.listChats(pid).then(r => setChats(r.data)).catch(() => {})
    } catch (err: unknown) {
      if (!(err instanceof Error && err.name === "AbortError")) {
        setMessages(m => {
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

  // ── Render ───────────────────────────────────────────────────────────────

  const hasAttachments = attachedFiles.length > 0 || reusedDocs.length > 0 || selectedPrompts.length > 0

  return (
    <div className="flex h-[calc(100vh-2rem)] gap-0 overflow-hidden">

      {/* ── Column 1: Chat list ────────────────────────────────────────────── */}
      <div className="w-52 shrink-0 flex flex-col border-r bg-muted/20 overflow-hidden">
        <div className="flex items-center justify-between px-3 py-2 border-b">
          <div className="flex items-center gap-1.5 min-w-0">
            <FolderOpen className="h-4 w-4 shrink-0 text-primary" />
            <span className="text-sm font-semibold truncate">{projectTitle || t("agent.project")}</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0 text-destructive hover:text-destructive"
            title={t("agent.delete_project")}
            onClick={() => setConfirmDeleteProject(true)}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto py-1">
          <button
            onClick={handleNewChat}
            className="flex items-center gap-2 w-full px-3 py-1.5 text-xs text-muted-foreground hover:bg-muted transition-colors"
          >
            <Plus className="h-3.5 w-3.5" />
            {t("agent.new_chat")}
          </button>

          {chats.map(chat => (
            <div
              key={chat.id}
              className={cn(
                "group flex items-center gap-1 px-3 py-1.5 cursor-pointer transition-colors text-sm",
                activeChatId === chat.id ? "bg-primary/10 text-primary font-medium" : "hover:bg-muted text-muted-foreground"
              )}
              onClick={() => selectChat(chat.id)}
            >
              <MessageSquare className="h-3.5 w-3.5 shrink-0" />
              <span className="flex-1 truncate text-xs">{chat.title}</span>
              <button
                className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 rounded hover:text-destructive"
                onClick={e => { e.stopPropagation(); setConfirmDeleteChat(chat.id) }}
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}

          {chats.length === 0 && (
            <p className="px-3 py-4 text-xs text-muted-foreground">{t("agent.no_chats")}</p>
          )}
        </div>
      </div>

      {/* ── Column 2: Chat area ───────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 p-4 gap-3 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-primary" />
            <h1 className="text-xl font-semibold">{t("agent.title")}</h1>
          </div>
          <div className="flex items-center gap-2">
            {agentModel && <Badge variant="secondary" className="text-xs font-mono">{agentModel}</Badge>}
            {online === false && (
              <Badge variant="destructive" className="gap-1 text-xs">
                <WifiOff className="h-3 w-3" />
                {t("agent.offline")}
              </Badge>
            )}
          </div>
        </div>

        {online === false && (
          <div className="rounded-md border border-destructive/40 bg-destructive/5 px-4 py-2 text-sm text-destructive shrink-0">
            {t("agent.offline_hint")}
          </div>
        )}

        {/* Suggested prompts */}
        {prompts.length > 0 && (
          <div className="flex flex-wrap gap-1.5 shrink-0">
            {prompts.map(p => (
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

        {/* Messages */}
        <div className="flex-1 overflow-y-auto rounded-lg border bg-muted/20 p-4 space-y-4 min-h-0">
          {messages.length === 0 && activeChatId === null && (
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
                      {msg.files.map(f => (
                        <Badge key={f} variant="outline" className="text-xs gap-1">
                          <Paperclip className="h-3 w-3" />{f}
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
                    const isLiveStreaming = streaming && i === messages.length - 1 && !toolLog
                    return (
                      <>
                        {isLiveStreaming && (
                          <div className="rounded border border-muted bg-muted/40 px-2 py-1.5 space-y-0.5 font-mono text-xs text-muted-foreground">
                            {msg.text.split("\n").filter(l => l.trim()).map((l, li) => (
                              <div key={li} className="flex items-start gap-1.5">
                                <span className="mt-px shrink-0 animate-pulse">›</span>
                                <span>{l.replace(/\*|`/g, "").trim()}</span>
                              </div>
                            ))}
                          </div>
                        )}
                        {!isLiveStreaming && (
                          <div className="prose prose-sm max-w-none dark:prose-invert">
                            <ReactMarkdown components={{ img: ({ src, alt }) => src ? <AuthImage src={src} alt={alt ?? ""} /> : null }}>
                              {main}
                            </ReactMarkdown>
                          </div>
                        )}
                        {toolLog && (
                          <div className="border-t pt-2 mt-2">
                            <button
                              onClick={() => toggleTools(i)}
                              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                            >
                              {expandedTools[i] ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
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
        <form onSubmit={handleSend} className="space-y-2 shrink-0">
          {hasAttachments && (
            <div className="flex flex-wrap gap-1.5">
              {selectedPrompts.map(p => (
                <Badge key={p.label} variant="default" className="gap-1 text-xs pr-1">
                  {p.label}
                  <button type="button" onClick={() => togglePrompt(p)} className="ml-0.5 rounded-full hover:bg-primary-foreground/20 p-0.5">
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              ))}
              {reusedDocs.map(d => (
                <Badge key={d.id} variant="secondary" className="gap-1 text-xs pr-1">
                  <FileText className="h-3 w-3" />{d.filename}
                  <button type="button" onClick={() => removeReusedDoc(d.id)} className="ml-0.5 rounded-full hover:bg-muted-foreground/20 p-0.5">
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              ))}
              {attachedFiles.map((f, i) => (
                <Badge key={i} variant="secondary" className="gap-1 text-xs pr-1">
                  <Paperclip className="h-3 w-3" />{f.name}
                  <button type="button" onClick={() => removeFile(i)} className="ml-0.5 rounded-full hover:bg-muted-foreground/20 p-0.5">
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              ))}
            </div>
          )}

          <div className="flex gap-2 items-end">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button type="button" variant="outline" size="icon" className="shrink-0 h-9 w-9" disabled={streaming || !online} title={t("agent.actions_menu")}>
                  <Plus className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-56 max-h-72 overflow-y-auto bg-white dark:bg-zinc-900">
                <DropdownMenuItem onSelect={() => fileRef.current?.click()}>
                  <Paperclip className="h-4 w-4 mr-2" />{t("agent.attach_files")}
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onSelect={() => { setMessages([]); setInput(""); setAttachedFiles([]); setReusedDocs([]); setSelectedPrompts([]) }}>
                  <Trash2 className="h-4 w-4 mr-2" />{t("agent.new_conversation")}
                </DropdownMenuItem>
                {prompts.length > 0 && (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuLabel>{t("agent.analyze_document")}</DropdownMenuLabel>
                    {prompts.map(p => (
                      <DropdownMenuItem key={p.label} onSelect={() => togglePrompt(p)}>
                        <Check className={cn("h-4 w-4 mr-2", selectedPrompts.some(x => x.label === p.label) ? "opacity-100" : "opacity-0")} />
                        {p.label}
                      </DropdownMenuItem>
                    ))}
                  </>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
            <input ref={fileRef} type="file" multiple accept="image/*,.pdf" className="hidden" onChange={handleFileChange} />

            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={online === false ? t("agent.offline") : t("agent.placeholder")}
              disabled={!online || streaming}
              rows={1}
              className="flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50 overflow-hidden"
            />

            {streaming ? (
              <Button type="button" size="icon" variant="destructive" className="shrink-0 h-9 w-9" onClick={() => abortRef.current?.abort()}>
                <Square className="h-4 w-4" />
              </Button>
            ) : (
              <Button type="submit" size="icon" className="shrink-0 h-9 w-9" disabled={!online || (!input.trim() && selectedPrompts.length === 0)}>
                <Send className="h-4 w-4" />
              </Button>
            )}
          </div>
          <p className="text-xs text-muted-foreground px-1">{t("agent.input_hint")}</p>
        </form>
      </div>

      {/* ── Column 3: Documents ───────────────────────────────────────────── */}
      <div className="w-56 shrink-0 flex flex-col border-l bg-muted/20 overflow-hidden">
        <div className="px-3 py-2 border-b">
          <span className="text-sm font-semibold">{t("agent.project_documents")}</span>
        </div>
        <div className="flex-1 overflow-y-auto py-1">
          {documents.length === 0 && (
            <p className="px-3 py-4 text-xs text-muted-foreground">{t("agent.no_documents")}</p>
          )}
          {documents.map(doc => (
            <div
              key={doc.id}
              className={cn(
                "group flex items-center gap-2 px-3 py-1.5 cursor-pointer transition-colors",
                reusedDocs.some(d => d.id === doc.id) ? "bg-primary/10" : "hover:bg-muted"
              )}
              onClick={() => toggleReusedDoc(doc)}
              title={`${doc.filename} (${formatBytes(doc.size_bytes)})`}
            >
              <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
              <div className="flex-1 min-w-0">
                <p className="text-xs truncate">{doc.filename}</p>
                <p className="text-xs text-muted-foreground">{formatBytes(doc.size_bytes)}</p>
              </div>
              {reusedDocs.some(d => d.id === doc.id) && (
                <Check className="h-3.5 w-3.5 shrink-0 text-primary" />
              )}
              <button
                className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 rounded hover:text-destructive"
                onClick={e => { e.stopPropagation(); handleDeleteDocument(doc.id) }}
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
        <div className="px-3 py-2 border-t text-xs text-muted-foreground">
          {t("agent.click_to_reuse")}
        </div>
      </div>

      {/* ── Confirm delete project ─────────────────────────────────────────── */}
      {confirmDeleteProject && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="bg-white dark:bg-zinc-900 rounded-lg border shadow-lg p-6 max-w-sm w-full mx-4 space-y-4">
            <div className="flex items-center gap-2 text-destructive">
              <AlertTriangle className="h-5 w-5" />
              <h2 className="font-semibold">{t("agent.delete_project_confirm_title")}</h2>
            </div>
            <p className="text-sm text-muted-foreground">{t("agent.delete_project_confirm_body")}</p>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setConfirmDeleteProject(false)}>{t("common.cancel")}</Button>
              <Button variant="destructive" onClick={handleDeleteProject}>{t("common.delete")}</Button>
            </div>
          </div>
        </div>
      )}

      {/* ── Confirm delete chat ────────────────────────────────────────────── */}
      {confirmDeleteChat !== null && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="bg-white dark:bg-zinc-900 rounded-lg border shadow-lg p-6 max-w-sm w-full mx-4 space-y-4">
            <div className="flex items-center gap-2 text-destructive">
              <AlertTriangle className="h-5 w-5" />
              <h2 className="font-semibold">{t("agent.delete_chat_confirm_title")}</h2>
            </div>
            <p className="text-sm text-muted-foreground">{t("agent.delete_chat_confirm_body")}</p>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setConfirmDeleteChat(null)}>{t("common.cancel")}</Button>
              <Button variant="destructive" onClick={() => handleDeleteChat(confirmDeleteChat)}>{t("common.delete")}</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
