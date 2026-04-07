import { NavLink, useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { FolderOpen, MessageSquare, Users, Microscope, LogOut, Globe, ClipboardCheck, Bot, Plus, ChevronDown, ChevronRight, Settings, HelpCircle, ChevronsUpDown } from "lucide-react"
import { cn } from "@/lib/utils"
import { useAuthStore } from "@/store/auth"
import { authApi, ragApi, usersApi, agentProjectsApi, type AgentProject } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator } from "@/components/ui/dropdown-menu"
import SettingsDialog from "@/components/settings/SettingsDialog"
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

function getInitials(name: string | undefined): string {
  if (!name) return "?"
  return name
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map(w => w[0].toUpperCase())
    .join("")
}

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

  // Settings dialog
  const [settingsOpen, setSettingsOpen] = useState(false)

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

    usersApi.getSettings().then(async r => {
      const s = r.data
      setOpenaiKeyConfigured(s.openai_key_configured)

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

  const initials = getInitials(user?.full_name)

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

        {/* Agent projects sub-section */}
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

      {/* Footer — ChatGPT-style user menu */}
      <div className="border-t pt-2 px-1">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="flex items-center gap-2.5 w-full rounded-md px-2 py-2 hover:bg-muted transition-colors">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-semibold">
                {initials}
              </span>
              <span className="flex-1 text-left text-sm font-medium truncate">{user?.full_name}</span>
              <ChevronsUpDown className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent side="top" align="start" className="w-52">
            <DropdownMenuItem onClick={() => setSettingsOpen(true)}>
              <Settings className="mr-2 h-4 w-4" />
              {t("nav.settings")}
            </DropdownMenuItem>
            <DropdownMenuItem disabled>
              <HelpCircle className="mr-2 h-4 w-4" />
              {t("nav.help")}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={toggleLang}>
              <Globe className="mr-2 h-4 w-4" />
              {i18n.language === "it" ? "English" : "Italiano"}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={handleLogout} className="text-destructive focus:text-destructive">
              <LogOut className="mr-2 h-4 w-4" />
              {t("auth.logout")}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        <SettingsDialog
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          models={models}
          ollamaUp={ollamaUp}
          refreshing={refreshing}
          onRefresh={loadModels}
          openaiModels={openaiModels}
          currentModel={currentModel}
          onModelChange={handleModelChange}
          currentOcrModel={currentOcrModel}
          onOcrModelChange={handleOcrModelChange}
          currentVlmModel={currentVlmModel}
          onVlmModelChange={handleVlmModelChange}
          openaiKeyConfigured={openaiKeyConfigured}
          currentEmbedModel={currentEmbedModel}
          onEmbedModelChange={handleEmbedModelChange}
          showKeyInput={showKeyInput}
          setShowKeyInput={setShowKeyInput}
          keyDraft={keyDraft}
          setKeyDraft={setKeyDraft}
          savingKey={savingKey}
          keyError={keyError}
          onSaveKey={handleSaveKey}
          onDeleteKey={handleDeleteKey}
        />
      </div>
    </aside>
  )
}
