import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { Users, ClipboardList, Plus, Trash2, RefreshCw, ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { api, usersApi, auditApi, type User, type AuditLogEntry } from "@/lib/api"

const AUDIT_ACTIONS = [
  "login",
  "project_create",
  "project_delete",
  "document_upload",
  "document_delete",
  "analysis_run",
  "analysis_clear",
  "pdf_download",
]

const ACTION_COLORS: Record<string, string> = {
  login: "bg-blue-100 text-blue-800",
  project_create: "bg-green-100 text-green-800",
  project_delete: "bg-red-100 text-red-800",
  document_upload: "bg-green-100 text-green-800",
  document_delete: "bg-red-100 text-red-800",
  analysis_run: "bg-purple-100 text-purple-800",
  analysis_clear: "bg-orange-100 text-orange-800",
  pdf_download: "bg-gray-100 text-gray-800",
}

export default function AdminPage() {
  const { t } = useTranslation()
  const [tab, setTab] = useState<"users" | "audit">("users")

  // ── Users state ──────────────────────────────────────────────────────────────
  const [users, setUsers] = useState<User[]>([])
  const [newEmail, setNewEmail] = useState("")
  const [newName, setNewName] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [newRole, setNewRole] = useState<"admin" | "examiner" | "viewer">("examiner")
  const [userError, setUserError] = useState("")

  async function loadUsers() {
    const { data } = await usersApi.list()
    setUsers(data)
  }

  async function handleCreateUser(e: React.FormEvent) {
    e.preventDefault()
    setUserError("")
    try {
      const { data } = await usersApi.create({ email: newEmail, full_name: newName, password: newPassword, role: newRole })
      setUsers((u) => [...u, data])
      setNewEmail(""); setNewName(""); setNewPassword(""); setNewRole("examiner")
    } catch (err: any) {
      setUserError(err?.response?.data?.detail ?? t("common.error"))
    }
  }

  async function handleDeactivate(userId: number) {
    if (!confirm(t("common.confirm"))) return
    await usersApi.deactivate(userId)
    setUsers((u) => u.map((x) => x.id === userId ? { ...x, is_active: false } : x))
  }

  // ── Audit state ──────────────────────────────────────────────────────────────
  const [auditItems, setAuditItems] = useState<AuditLogEntry[]>([])
  const [auditTotal, setAuditTotal] = useState(0)
  const [auditPage, setAuditPage] = useState(1)
  const [filterAction, setFilterAction] = useState("")
  const [filterEmail, setFilterEmail] = useState("")
  const PAGE_SIZE = 50

  async function loadAudit(page = auditPage) {
    const { data } = await auditApi.list(page, PAGE_SIZE, filterAction || undefined, filterEmail || undefined)
    setAuditItems(data.items)
    setAuditTotal(data.total)
  }

  useEffect(() => { loadUsers() }, [])
  useEffect(() => { if (tab === "audit") loadAudit(1) }, [tab])

  function handleAuditFilter(e: React.FormEvent) {
    e.preventDefault()
    setAuditPage(1)
    loadAudit(1)
  }

  function changePage(delta: number) {
    const next = auditPage + delta
    setAuditPage(next)
    loadAudit(next)
  }

  const totalPages = Math.max(1, Math.ceil(auditTotal / PAGE_SIZE))

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-4">
      <h1 className="text-xl font-semibold">{t("nav.admin")}</h1>

      {/* Tab bar */}
      <div className="flex gap-1 border-b pb-0">
        <button
          className={`flex items-center gap-1.5 px-4 py-2 text-sm font-medium border-b-2 transition-colors ${tab === "users" ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"}`}
          onClick={() => setTab("users")}
        >
          <Users className="h-4 w-4" />
          {t("admin.users")}
        </button>
        <button
          className={`flex items-center gap-1.5 px-4 py-2 text-sm font-medium border-b-2 transition-colors ${tab === "audit" ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"}`}
          onClick={() => setTab("audit")}
        >
          <ClipboardList className="h-4 w-4" />
          {t("admin.audit")}
        </button>
      </div>

      {/* ── Users tab ─────────────────────────────────────────────────────────── */}
      {tab === "users" && (
        <div className="space-y-4">
          {/* Create user form */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Plus className="h-4 w-4" />
                {t("admin.new_user")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleCreateUser} className="flex flex-wrap gap-2 items-end">
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-muted-foreground">{t("admin.email")}</label>
                  <Input className="h-8 text-sm w-48" type="email" value={newEmail} onChange={(e) => setNewEmail(e.target.value)} required />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-muted-foreground">{t("admin.full_name")}</label>
                  <Input className="h-8 text-sm w-40" value={newName} onChange={(e) => setNewName(e.target.value)} required />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-muted-foreground">Password</label>
                  <Input className="h-8 text-sm w-32" type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} required />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-muted-foreground">{t("admin.role")}</label>
                  <select className="h-8 text-sm border rounded-md px-2" value={newRole} onChange={(e) => setNewRole(e.target.value as any)}>
                    <option value="examiner">{t("admin.roles.examiner")}</option>
                    <option value="viewer">{t("admin.roles.viewer")}</option>
                    <option value="admin">{t("admin.roles.admin")}</option>
                  </select>
                </div>
                <Button type="submit" size="sm" className="h-8">{t("admin.new_user")}</Button>
              </form>
              {userError && <p className="text-xs text-destructive mt-2">{userError}</p>}
            </CardContent>
          </Card>

          {/* Users table */}
          <Card>
            <CardContent className="pt-4">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-xs text-muted-foreground">
                    <th className="text-left pb-2 font-medium">{t("admin.full_name")}</th>
                    <th className="text-left pb-2 font-medium">{t("admin.email")}</th>
                    <th className="text-left pb-2 font-medium">{t("admin.role")}</th>
                    <th className="text-left pb-2 font-medium">{t("admin.status")}</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {users.map((u) => (
                    <tr key={u.id} className="border-b last:border-0">
                      <td className="py-2">{u.full_name}</td>
                      <td className="py-2 text-muted-foreground">{u.email}</td>
                      <td className="py-2">
                        <Badge variant="outline" className="text-xs">{t(`admin.roles.${u.role}`)}</Badge>
                      </td>
                      <td className="py-2">
                        <Badge variant={u.is_active ? "default" : "secondary"} className="text-xs">
                          {u.is_active ? t("admin.active") : t("admin.inactive")}
                        </Badge>
                      </td>
                      <td className="py-2 text-right">
                        {u.is_active && (
                          <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground hover:text-destructive"
                            onClick={() => handleDeactivate(u.id)}>
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </div>
      )}

      {/* ── Audit tab ─────────────────────────────────────────────────────────── */}
      {tab === "audit" && (
        <div className="space-y-3">
          {/* Filters */}
          <form onSubmit={handleAuditFilter} className="flex flex-wrap gap-2 items-end">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">{t("admin.audit_filter_action")}</label>
              <select className="h-8 text-sm border rounded-md px-2 w-44"
                value={filterAction} onChange={(e) => setFilterAction(e.target.value)}>
                <option value="">{t("admin.audit_all_actions")}</option>
                {AUDIT_ACTIONS.map((a) => (
                  <option key={a} value={a}>{t(`admin.audit_actions.${a}`)}</option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">{t("admin.audit_filter_user")}</label>
              <Input className="h-8 text-sm w-48" placeholder="email…"
                value={filterEmail} onChange={(e) => setFilterEmail(e.target.value)} />
            </div>
            <Button type="submit" size="sm" className="h-8 gap-1">
              <RefreshCw className="h-3.5 w-3.5" />
              {t("common.confirm")}
            </Button>
          </form>

          {/* Summary */}
          {auditTotal > 0 && (
            <p className="text-xs text-muted-foreground">
              {t("admin.audit_page", { page: auditPage, total: auditTotal })}
            </p>
          )}

          {/* Table */}
          <Card>
            <CardContent className="pt-4 overflow-x-auto">
              {auditItems.length === 0 ? (
                <p className="text-sm text-muted-foreground">{t("admin.audit_empty")}</p>
              ) : (
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b text-muted-foreground">
                      <th className="text-left pb-2 font-medium">{t("admin.audit_timestamp")}</th>
                      <th className="text-left pb-2 font-medium">{t("admin.audit_user")}</th>
                      <th className="text-left pb-2 font-medium">{t("admin.audit_action")}</th>
                      <th className="text-left pb-2 font-medium">{t("admin.audit_resource")}</th>
                      <th className="text-left pb-2 font-medium">{t("admin.audit_detail")}</th>
                      <th className="text-left pb-2 font-medium">{t("admin.audit_ip")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {auditItems.map((entry) => (
                      <tr key={entry.id} className="border-b last:border-0">
                        <td className="py-1.5 pr-3 whitespace-nowrap text-muted-foreground">
                          {new Date(entry.timestamp).toLocaleString()}
                        </td>
                        <td className="py-1.5 pr-3">{entry.user_email}</td>
                        <td className="py-1.5 pr-3">
                          <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${ACTION_COLORS[entry.action] ?? "bg-gray-100 text-gray-800"}`}>
                            {t(`admin.audit_actions.${entry.action}`, { defaultValue: entry.action })}
                          </span>
                        </td>
                        <td className="py-1.5 pr-3 text-muted-foreground">
                          {entry.resource_type ? `${entry.resource_type}/${entry.resource_id}` : "—"}
                        </td>
                        <td className="py-1.5 pr-3">{entry.detail ?? "—"}</td>
                        <td className="py-1.5 text-muted-foreground">{entry.ip_address ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </CardContent>
          </Card>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center gap-2 justify-end">
              <Button variant="outline" size="sm" className="h-7 gap-1"
                disabled={auditPage <= 1} onClick={() => changePage(-1)}>
                <ChevronLeft className="h-3.5 w-3.5" />
                {t("admin.audit_prev")}
              </Button>
              <span className="text-xs text-muted-foreground">{auditPage} / {totalPages}</span>
              <Button variant="outline" size="sm" className="h-7 gap-1"
                disabled={auditPage >= totalPages} onClick={() => changePage(1)}>
                {t("admin.audit_next")}
                <ChevronRight className="h-3.5 w-3.5" />
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
