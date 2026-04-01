import { useEffect, useRef, useState } from "react"
import { useParams, useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { ArrowLeft, Upload, Play, Trash2, FileText, ChevronDown, ChevronUp, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { analysisApi, projectsApi, type Analysis, type Document, type Project } from "@/lib/api"

const ANALYSIS_TYPES = [
  "htr",
  "signature_detection",
  "signature_verification",
  "ner",
  "writer_identification",
  "graphology",
  "pipeline",
  "dating",
]

function AnalysisCard({ analysis }: { analysis: Analysis }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)

  return (
    <Card>
      <CardHeader className="py-3 px-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">
            {t(`project.analysis_types.${analysis.analysis_type}`)}
          </CardTitle>
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setOpen((o) => !o)}>
            {open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </Button>
        </div>
      </CardHeader>
      {open && analysis.result_text && (
        <CardContent className="pt-0 px-4 pb-4">
          <pre className="text-xs bg-muted rounded p-3 whitespace-pre-wrap overflow-auto max-h-80">
            {analysis.result_text}
          </pre>
        </CardContent>
      )}
    </Card>
  )
}

export default function ProjectDetailPage() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)
  const { t } = useTranslation()
  const navigate = useNavigate()

  const [project, setProject] = useState<Project | null>(null)
  const [documents, setDocuments] = useState<Document[]>([])
  const [analyses, setAnalyses] = useState<Analysis[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedDoc, setSelectedDoc] = useState<number | null>(null)
  const [runningType, setRunningType] = useState<string | null>(null)
  const [refDoc, setRefDoc] = useState<number | null>(null)
  const [showRefPicker, setShowRefPicker] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  async function load() {
    const [{ data: proj }, { data: docs }, { data: ans }] = await Promise.all([
      projectsApi.get(projectId),
      projectsApi.listDocuments(projectId),
      analysisApi.list(projectId),
    ])
    setProject(proj)
    setDocuments(docs)
    setAnalyses(ans)
    if (docs.length > 0 && !selectedDoc) setSelectedDoc(docs[0].id)
    setLoading(false)
  }

  useEffect(() => { load() }, [projectId])

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const { data } = await projectsApi.uploadDocument(projectId, file)
    setDocuments((d) => [...d, data])
    setSelectedDoc(data.id)
    e.target.value = ""
  }

  async function handleDeleteDoc(docId: number) {
    if (!confirm(t("common.confirm"))) return
    await projectsApi.deleteDocument(projectId, docId)
    setDocuments((d) => d.filter((x) => x.id !== docId))
    if (selectedDoc === docId) setSelectedDoc(documents.find((x) => x.id !== docId)?.id ?? null)
  }

  async function handleRunAnalysis(type: string) {
    if (!selectedDoc) return
    if (type === "signature_verification") {
      setShowRefPicker(true)
      return
    }
    setRunningType(type)
    try {
      const { data } = await analysisApi.run(type, projectId, selectedDoc)
      setAnalyses((a) => [data, ...a])
    } finally {
      setRunningType(null)
    }
  }

  async function handleRunSigVerify() {
    if (!selectedDoc || !refDoc) return
    setShowRefPicker(false)
    setRunningType("signature_verification")
    try {
      const { data } = await analysisApi.runSignatureVerification(projectId, selectedDoc, refDoc)
      setAnalyses((a) => [data, ...a])
    } finally {
      setRunningType(null)
    }
  }

  async function handleClearAnalyses() {
    if (!confirm(t("project.clear_analyses_confirm"))) return
    await analysisApi.clearAll(projectId)
    setAnalyses([])
  }

  if (loading) return <div className="p-6 text-muted-foreground">{t("common.loading")}</div>
  if (!project) return <div className="p-6">{t("common.error")}</div>

  return (
    <>
    {showRefPicker && (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-background rounded-lg p-6 w-96 space-y-4 shadow-xl">
          <h2 className="font-semibold">{t("project.select_reference_doc")}</h2>
          <p className="text-sm text-muted-foreground">{t("project.select_reference_doc_hint")}</p>
          <select
            className="w-full border rounded-md px-3 py-2 text-sm"
            value={refDoc ?? ""}
            onChange={(e) => setRefDoc(Number(e.target.value))}
          >
            <option value="">{t("project.select_reference_placeholder")}</option>
            {documents.filter((d) => d.id !== selectedDoc).map((d) => (
              <option key={d.id} value={d.id}>{d.filename}</option>
            ))}
          </select>
          <div className="flex gap-2 justify-end">
            <Button variant="outline" onClick={() => setShowRefPicker(false)}>{t("common.cancel")}</Button>
            <Button disabled={!refDoc} onClick={handleRunSigVerify}>{t("project.run_analysis")}</Button>
          </div>
        </div>
      </div>
    )}
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="icon" onClick={() => navigate("/projects")}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <div>
          <h1 className="text-xl font-semibold">{project.title}</h1>
          {project.description && (
            <p className="text-sm text-muted-foreground">{project.description}</p>
          )}
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Left column: documents + analysis runner */}
        <div className="space-y-4">
          {/* Documents */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">{t("project.documents")}</CardTitle>
                <Button variant="outline" size="sm" className="h-7 text-xs gap-1" onClick={() => fileRef.current?.click()}>
                  <Upload className="h-3.5 w-3.5" />
                  {t("project.upload")}
                </Button>
                <input ref={fileRef} type="file" accept="image/*,.pdf" className="hidden" onChange={handleUpload} />
              </div>
            </CardHeader>
            <CardContent className="space-y-1">
              {documents.length === 0 ? (
                <p className="text-xs text-muted-foreground">{t("project.no_documents")}</p>
              ) : (
                documents.map((doc) => (
                  <div
                    key={doc.id}
                    className={`flex items-center gap-2 rounded-md px-2 py-1.5 cursor-pointer text-sm transition-colors ${
                      selectedDoc === doc.id ? "bg-primary/10 text-primary" : "hover:bg-muted"
                    }`}
                    onClick={() => setSelectedDoc(doc.id)}
                  >
                    <FileText className="h-3.5 w-3.5 shrink-0" />
                    <span className="truncate flex-1">{doc.filename}</span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 text-muted-foreground hover:text-destructive shrink-0"
                      onClick={(e) => { e.stopPropagation(); handleDeleteDoc(doc.id) }}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                ))
              )}
            </CardContent>
          </Card>

          {/* Analysis runner */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">{t("project.run_analysis")}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-1">
              {ANALYSIS_TYPES.map((type) => (
                <Button
                  key={type}
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start text-xs h-8"
                  disabled={!selectedDoc || runningType !== null}
                  onClick={() => handleRunAnalysis(type)}
                >
                  {runningType === type ? (
                    <span className="text-muted-foreground">{t("project.running")}</span>
                  ) : (
                    <>
                      <Play className="h-3 w-3 mr-2" />
                      {t(`project.analysis_types.${type}`)}
                    </>
                  )}
                </Button>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Right column: analyses results */}
        <div className="lg:col-span-2 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-muted-foreground">{t("project.analyses")}</h2>
            {analyses.length > 0 && (
              <Button variant="ghost" size="sm" className="h-7 text-xs text-destructive hover:text-destructive gap-1" onClick={handleClearAnalyses}>
                <X className="h-3 w-3" />
                {t("project.clear_analyses")}
              </Button>
            )}
          </div>
          {analyses.length === 0 ? (
            <p className="text-sm text-muted-foreground">{t("project.no_analyses")}</p>
          ) : (
            analyses.map((a) => <AnalysisCard key={a.id} analysis={a} />)
          )}
        </div>
      </div>
    </div>
    </>
  )
}
