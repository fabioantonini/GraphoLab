import { NavLink, useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { FolderOpen, MessageSquare, Users, Microscope, LogOut, Globe, ClipboardCheck, RefreshCw, Bot, Plus, ChevronDown, ChevronRight, Check, Trash2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { useAuthStore } from "@/store/auth"
import { authApi, ragApi, usersApi, agentProjectsApi, type AgentProject } from "@/lib/api"
import { Button } from "@/components/ui/button"
import i18n from "@/i18n"
import { useState, useEffect } from "react"

const links = [
  { to: "/projects", label: "nav.projects", icon: FolderOpen },
  { to: "/agent", label: "nav.agent", icon: Bot },
  { to: "/rag", label: "nav.rag", icon: MessageSquare },
  { to: "/compliance", label: "nav.compliance", icon: ClipboardCheck },
]

const adminLinks = [
  { to: "/admin/users", label: "nav.admin", icon: Users },
]

export default function Sidebar() {
  const { t } = useTranslation()
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()

  const [agentProjects, setAgentProjects] = useState<AgentProject[]>([])
  const [agentExpanded, setAgentExpanded] = useState(true)
  const [newProjectTitle, setNewProjectTitle] = useState("")
  const [creatingProject, setCreatingProject] = useState(false)
  const [showNewProjectInput, setShowNewProjectInput] = useState(false)

  const [models, setModels] = useState<string[]>([])
  const [currentModel, setCurrentModel] = useState<string>("")
  const [currentOcrModel, setCurrentOcrModel] = useState<string>("easyocr")
  const [currentVlmModel, setCurrentVlmModel] = useState<string>("qwen3-vl:8b")
  const [ollamaUp, setOllamaUp] = useState<boolean | null>(null)
  const [refreshing, setRefreshing] = useState(false)

  // OpenAI state
  const [openaiModels, setOpenaiModels] = useState<{ llm: string[]; vlm: string[]; embed: string[] }>({ llm: [], vlm: [], embed: [] })
  const [currentEmbedModel, setCurrentEmbedModel] = useState<string>("nomic-embed-text")
  const [openaiKeyConfigured, setOpenaiKeyConfigured] = useState(false)
  const [showKeyInput, setShowKeyInput] = useState(false)
  const [keyDraft, setKeyDraft] = useState("")
  const [savingKey, setSavingKey] = useState(false)
  const [keyError, setKeyError] = useState("")

  const OCR_MODELS = ["easyocr", "vlm", "paddleocr", "trocr"]

  async function loadModels() {
    setRefreshing(true)
    try {
      const r = await ragApi.status()
      setOllamaUp(r.data.ollama_reachable)
      setModels(r.data.models)
      setOpenaiModels({
        llm:   r.data.openai_llm_models   ?? [],
        vlm:   r.data.openai_vlm_models   ?? [],
        embed: r.data.openai_embed_models ?? [],
      })
    } catch {
      setOllamaUp(false)
    } finally {
      setRefreshing(false)
    }
  }

  useEffect(() => {
    loadModels()
    agentProjectsApi.listProjects().then(r => setAgentProjects(r.data)).catch(() => {})

    // Restore per-user settings (model preferences + OpenAI key status) from DB.
    // If the user has saved preferences, apply them to the backend global state so
    // all subsequent requests use the correct model.
    usersApi.getSettings().then(async r => {
      const s = r.data
      setOpenaiKeyConfigured(s.openai_key_configured)

      // Restore saved models — fall back to current server state if not saved yet
      if (s.rag_model) {
        setCurrentModel(s.rag_model)
        await ragApi.setModel(s.rag_model).catch(() => {})
      } else {
        ragApi.getModel().then(r2 => setCurrentModel(r2.data.model)).catch(() => {})
      }

      if (s.vlm_model) {
        setCurrentVlmModel(s.vlm_model)
        await ragApi.setVlmModel(s.vlm_model).catch(() => {})
      } else {
        ragApi.getVlmModel().then(r2 => setCurrentVlmModel(r2.data.vlm_model)).catch(() => {})
      }

      if (s.ocr_model) {
        setCurrentOcrModel(s.ocr_model)
        await ragApi.setOcrModel(s.ocr_model).catch(() => {})
      } else {
        ragApi.getOcrModel().then(r2 => setCurrentOcrModel(r2.data.ocr_model)).catch(() => {})
      }

      if (s.embed_model) {
        setCurrentEmbedModel(s.embed_model)
        await ragApi.setEmbedModel(s.embed_model).catch(() => {})
      } else {
        ragApi.getEmbedModel().then(r2 => setCurrentEmbedModel(r2.data.embed_model)).catch(() => {})
      }
    }).catch(() => {
      // Fallback: read current server state directly
      ragApi.getModel().then(r => setCurrentModel(r.data.model)).catch(() => {})
      ragApi.getOcrModel().then(r => setCurrentOcrModel(r.data.ocr_model)).catch(() => {})
      ragApi.getVlmModel().then(r => setCurrentVlmModel(r.data.vlm_model)).catch(() => {})
      ragApi.getEmbedModel().then(r => setCurrentEmbedModel(r.data.embed_model)).catch(() => {})
    })
  }, [])

  async function handleEmbedModelChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const model = e.target.value
    setCurrentEmbedModel(model)
    try { await ragApi.setEmbedModel(model) } catch { /* no-op */ }
    usersApi.saveModelPreferences({ embed_model: model }).catch(() => {})
  }

  async function handleSaveKey(e: React.FormEvent) {
    e.preventDefault()
    setKeyError("")
    setSavingKey(true)
    try {
      await usersApi.saveOpenAIKey(keyDraft)
      setOpenaiKeyConfigured(true)
      setShowKeyInput(false)
      setKeyDraft("")
      await loadModels()
    } catch {
      setKeyError(t("config.openai_key_error"))
    } finally {
      setSavingKey(false)
    }
  }

  async function handleDeleteKey() {
    try {
      await usersApi.deleteOpenAIKey()
      setOpenaiKeyConfigured(false)
      await loadModels()
    } catch {
      // no-op
    }
  }

  async function handleCreateProject() {
    if (!newProjectTitle.trim()) return
    setCreatingProject(true)
    try {
      const { data } = await agentProjectsApi.createProject(newProjectTitle.trim())
      setAgentProjects(prev => [data, ...prev])
      setNewProjectTitle("")
      setShowNewProjectInput(false)
      navigate(`/agent/projects/${data.id}`)
    } catch {
      // no-op
    } finally {
      setCreatingProject(false)
    }
  }

  async function handleModelChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const model = e.target.value
    setCurrentModel(model)
    try { await ragApi.setModel(model) } catch { /* no-op */ }
    usersApi.saveModelPreferences({ rag_model: model }).catch(() => {})
  }

  async function handleOcrModelChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const model = e.target.value
    setCurrentOcrModel(model)
    try { await ragApi.setOcrModel(model) } catch { /* no-op */ }
    usersApi.saveModelPreferences({ ocr_model: model }).catch(() => {})
  }

  async function handleVlmModelChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const model = e.target.value
    setCurrentVlmModel(model)
    try { await ragApi.setVlmModel(model) } catch { /* no-op */ }
    usersApi.saveModelPreferences({ vlm_model: model }).catch(() => {})
  }

  async function handleLogout() {
    try { await authApi.logout() } catch { /* no-op */ }
    logout()
  }

  function toggleLang() {
    i18n.changeLanguage(i18n.language === "it" ? "en" : "it")
  }

  return (
    <aside className="flex flex-col w-56 min-h-screen bg-card border-r px-3 py-4 gap-1">
      {/* Logo */}
      <div className="flex items-center gap-2 px-2 mb-6">
        <Microscope className="h-5 w-5 text-primary" />
        <span className="font-semibold text-base">{t("app.name")}</span>
      </div>

      {/* Main nav */}
      <nav className="flex-1 space-y-0.5 overflow-y-auto">
        {links.filter(l => l.to !== "/agent").slice(0, links.findIndex(l => l.to === "/agent")).map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )
            }
          >
            <Icon className="h-4 w-4" />
            {t(label)}
          </NavLink>
        ))}

        {/* Agente Documentale link */}
        <NavLink
          to="/agent"
          end
          className={({ isActive }) =>
            cn(
              "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
              isActive
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:bg-muted hover:text-foreground"
            )
          }
        >
          <Bot className="h-4 w-4" />
          {t("nav.agent")}
        </NavLink>

        {/* Agent projects sub-section — directly under Agente Documentale */}
        <div className="pt-1">
          <button
            onClick={() => setAgentExpanded(v => !v)}
            className="flex items-center gap-1.5 w-full px-3 py-1 text-xs font-semibold text-muted-foreground tracking-wide hover:text-foreground transition-colors"
          >
            {agentExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            {t("nav.agent_projects")}
            <button
              onClick={e => { e.stopPropagation(); setShowNewProjectInput(v => !v) }}
              className="ml-auto p-0.5 rounded hover:bg-muted"
              title={t("agent.new_project")}
            >
              <Plus className="h-3 w-3" />
            </button>
          </button>

          {showNewProjectInput && (
            <div className="px-2 py-1 flex gap-1">
              <input
                autoFocus
                value={newProjectTitle}
                onChange={e => setNewProjectTitle(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter") handleCreateProject(); if (e.key === "Escape") setShowNewProjectInput(false) }}
                placeholder={t("agent.project_name_placeholder")}
                className="flex-1 rounded border bg-background px-2 py-0.5 text-xs focus:outline-none focus:ring-1 focus:ring-ring"
              />
              <Button size="icon" className="h-6 w-6 shrink-0" disabled={creatingProject} onClick={handleCreateProject}>
                <Plus className="h-3 w-3" />
              </Button>
            </div>
          )}

          {agentExpanded && agentProjects.map(p => (
            <NavLink
              key={p.id}
              to={`/agent/projects/${p.id}`}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-2 pl-6 pr-3 py-1.5 text-xs rounded-md transition-colors",
                  isActive ? "bg-primary/10 text-primary font-medium" : "text-muted-foreground hover:bg-muted hover:text-foreground"
                )
              }
            >
              <FolderOpen className="h-3.5 w-3.5 shrink-0" />
              <span className="truncate">{p.title}</span>
            </NavLink>
          ))}

          {agentExpanded && agentProjects.length === 0 && (
            <p className="pl-6 pr-3 py-1 text-xs text-muted-foreground">{t("agent.no_projects")}</p>
          )}
        </div>

        {/* Remaining main nav links (RAG, Compliance, …) */}
        {links.filter(l => l.to !== "/agent").slice(links.findIndex(l => l.to === "/agent")).map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )
            }
          >
            <Icon className="h-4 w-4" />
            {t(label)}
          </NavLink>
        ))}

        {user?.role === "admin" && (
          <>
            <div className="my-2 border-t" />
            {adminLinks.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  )
                }
              >
                <Icon className="h-4 w-4" />
                {t(label)}
              </NavLink>
            ))}
          </>
        )}
      </nav>

      {/* Configurazione */}
      <div className="border-t pt-3 px-3 pb-1">
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs font-semibold text-muted-foreground tracking-wide">
            {t("config.title")}
          </p>
          <button
            onClick={loadModels}
            disabled={refreshing}
            title={t("config.refresh")}
            className="text-muted-foreground hover:text-foreground disabled:opacity-40 transition-colors"
          >
            <RefreshCw className={`h-3 w-3 ${refreshing ? "animate-spin" : ""}`} />
          </button>
        </div>
        <label className="block text-xs text-muted-foreground mb-1">{t("config.model_label")}</label>
        {ollamaUp === false && openaiModels.llm.length === 0 ? (
          <p className="text-xs text-destructive">{t("config.model_offline")}</p>
        ) : models.length === 0 && openaiModels.llm.length === 0 ? (
          <p className="text-xs text-muted-foreground">{t("config.model_loading")}</p>
        ) : (
          <select
            value={currentModel}
            onChange={handleModelChange}
            className="w-full rounded-md border bg-background px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-ring"
          >
            {models.length > 0 && (
              <optgroup label="Ollama">
                {models.map(m => <option key={m} value={m}>{m}</option>)}
              </optgroup>
            )}
            {openaiModels.llm.length > 0 && (
              <optgroup label="OpenAI">
                {openaiModels.llm.map(m => <option key={m} value={m}>{m}</option>)}
              </optgroup>
            )}
          </select>
        )}
        <label className="block text-xs text-muted-foreground mt-2 mb-1">{t("config.ocr_model_label")}</label>
        <select
          value={currentOcrModel}
          onChange={handleOcrModelChange}
          className="w-full rounded-md border bg-background px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-ring"
        >
          {OCR_MODELS.map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <label className="block text-xs text-muted-foreground mt-2 mb-1">{t("config.vlm_model_label")}</label>
        {models.length === 0 && openaiModels.vlm.length === 0 ? (
          <p className="text-xs text-muted-foreground">{t("config.model_loading")}</p>
        ) : (
          <select
            value={currentVlmModel}
            onChange={handleVlmModelChange}
            className="w-full rounded-md border bg-background px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-ring"
          >
            {models.length > 0 && (
              <optgroup label="Ollama">
                {models.map(m => <option key={m} value={m}>{m}</option>)}
              </optgroup>
            )}
            {openaiModels.vlm.length > 0 && (
              <optgroup label="OpenAI">
                {openaiModels.vlm.map(m => <option key={m} value={m}>{m}</option>)}
              </optgroup>
            )}
          </select>
        )}

        {/* OpenAI Embedding model (visible only when OpenAI key is configured) */}
        {openaiKeyConfigured && (
          <>
            <label className="block text-xs text-muted-foreground mt-2 mb-1">{t("config.embed_model_label")}</label>
            <select
              value={currentEmbedModel}
              onChange={handleEmbedModelChange}
              className="w-full rounded-md border bg-background px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-ring"
            >
              <optgroup label="Ollama">
                <option value="nomic-embed-text">nomic-embed-text</option>
              </optgroup>
              <optgroup label="OpenAI">
                {openaiModels.embed.map(m => <option key={m} value={m}>{m}</option>)}
              </optgroup>
            </select>
          </>
        )}

        {/* OpenAI API Key section */}
        <div className="mt-3 pt-3 border-t">
          <label className="block text-xs font-semibold text-muted-foreground tracking-wide mb-1">
            {t("config.openai_key_label")}
          </label>
          {openaiKeyConfigured && !showKeyInput ? (
            <div className="flex items-center gap-2">
              <span className="text-xs text-green-600 dark:text-green-400">{t("config.openai_key_ok")}</span>
              <button
                className="text-xs text-muted-foreground underline"
                onClick={() => setShowKeyInput(true)}
              >
                {t("config.openai_key_change")}
              </button>
              <button
                className="text-xs text-destructive hover:opacity-70 transition-opacity ml-auto"
                onClick={handleDeleteKey}
                title={t("config.openai_key_remove")}
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </div>
          ) : !showKeyInput ? (
            <div>
              <p className="text-xs text-muted-foreground mb-1">{t("config.openai_key_missing")}</p>
              <button
                className="text-xs text-primary underline"
                onClick={() => setShowKeyInput(true)}
              >
                {t("config.openai_key_add")}
              </button>
            </div>
          ) : null}
          {showKeyInput && (
            <form onSubmit={handleSaveKey} className="flex gap-1 mt-1">
              <input
                type="password"
                value={keyDraft}
                onChange={e => { setKeyDraft(e.target.value); setKeyError("") }}
                placeholder="sk-…"
                className="flex-1 min-w-0 rounded border bg-background px-2 py-0.5 text-xs focus:outline-none focus:ring-1 focus:ring-ring"
              />
              <Button
                type="submit"
                size="icon"
                className="h-6 w-6 shrink-0"
                disabled={savingKey || !keyDraft.trim()}
              >
                <Check className="h-3 w-3" />
              </Button>
            </form>
          )}
          {keyError && <p className="text-xs text-destructive mt-1">{keyError}</p>}
        </div>
      </div>

      {/* Footer */}
      <div className="space-y-1 border-t pt-3">
        <div className="px-3 py-1.5 text-xs text-muted-foreground truncate">
          {user?.full_name}
          <br />
          <span className="opacity-70">{user?.email}</span>
        </div>
        <Button variant="ghost" size="sm" className="w-full justify-start gap-2 text-muted-foreground" onClick={toggleLang}>
          <Globe className="h-4 w-4" />
          {i18n.language === "it" ? "English" : "Italiano"}
        </Button>
        <Button variant="ghost" size="sm" className="w-full justify-start gap-2 text-muted-foreground" onClick={handleLogout}>
          <LogOut className="h-4 w-4" />
          {t("auth.logout")}
        </Button>
      </div>
    </aside>
  )
}
