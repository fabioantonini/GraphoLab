import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { Plus, Trash2, FolderOpen } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { projectsApi, type Project } from "@/lib/api"

const STATUS_VARIANT: Record<string, "default" | "secondary" | "success" | "warning" | "outline"> = {
  draft: "outline",
  in_progress: "warning",
  completed: "success",
  archived: "secondary",
}

export default function ProjectsPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [showCreate, setShowCreate] = useState(false)
  const [title, setTitle] = useState("")
  const [description, setDescription] = useState("")
  const [creating, setCreating] = useState(false)

  async function load() {
    try {
      const { data } = await projectsApi.list()
      setProjects(data)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault()
    setCreating(true)
    try {
      const { data } = await projectsApi.create({ title, description: description || undefined })
      setProjects((p) => [data, ...p])
      setTitle("")
      setDescription("")
      setShowCreate(false)
      navigate(`/projects/${data.id}`)
    } finally {
      setCreating(false)
    }
  }

  async function handleDelete(id: number, e: React.MouseEvent) {
    e.stopPropagation()
    if (!confirm(t("projects.delete_confirm"))) return
    await projectsApi.delete(id)
    setProjects((p) => p.filter((x) => x.id !== id))
  }

  return (
    <div className="p-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold">{t("projects.title")}</h1>
        <Button onClick={() => setShowCreate(true)}>
          <Plus className="h-4 w-4" />
          {t("projects.new")}
        </Button>
      </div>

      {/* Create form */}
      {showCreate && (
        <Card className="mb-6 border-primary/30">
          <CardHeader>
            <CardTitle className="text-base">{t("projects.create_title")}</CardTitle>
          </CardHeader>
          <form onSubmit={handleCreate}>
            <CardContent className="space-y-3">
              <div>
                <label className="text-sm font-medium">{t("projects.name_label")}</label>
                <Input
                  className="mt-1"
                  placeholder={t("projects.name_placeholder")}
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  required
                  autoFocus
                />
              </div>
              <div>
                <label className="text-sm font-medium">{t("projects.description_label")}</label>
                <Input
                  className="mt-1"
                  placeholder={t("projects.description_placeholder")}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </div>
            </CardContent>
            <CardFooter className="gap-2">
              <Button type="submit" size="sm" disabled={creating}>
                {t("projects.create")}
              </Button>
              <Button type="button" variant="outline" size="sm" onClick={() => setShowCreate(false)}>
                {t("projects.cancel")}
              </Button>
            </CardFooter>
          </form>
        </Card>
      )}

      {/* List */}
      {loading ? (
        <p className="text-muted-foreground">{t("common.loading")}</p>
      ) : projects.length === 0 ? (
        <div className="text-center py-20 text-muted-foreground">
          <FolderOpen className="h-12 w-12 mx-auto mb-3 opacity-30" />
          <p>{t("projects.empty")}</p>
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {projects.map((p) => (
            <Card
              key={p.id}
              className="cursor-pointer hover:shadow-md transition-shadow"
              onClick={() => navigate(`/projects/${p.id}`)}
            >
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between gap-2">
                  <CardTitle className="text-base leading-snug">{p.title}</CardTitle>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 shrink-0 text-muted-foreground hover:text-destructive"
                    onClick={(e) => handleDelete(p.id, e)}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
                {p.description && (
                  <CardDescription className="line-clamp-2">{p.description}</CardDescription>
                )}
              </CardHeader>
              <CardFooter className="pt-0 gap-2">
                <Badge variant={STATUS_VARIANT[p.status] ?? "outline"}>
                  {t(`projects.status.${p.status}`)}
                </Badge>
                <span className="text-xs text-muted-foreground ml-auto">
                  {t(p.document_count === 1 ? "projects.documents" : "projects.documents_plural", {
                    count: p.document_count,
                  })}
                </span>
              </CardFooter>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
