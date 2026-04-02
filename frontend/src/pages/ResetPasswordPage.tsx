import { useState } from "react"
import { useTranslation } from "react-i18next"
import { useSearchParams, useNavigate } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { authApi } from "@/lib/api"

export default function ResetPasswordPage() {
  const { t } = useTranslation()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const token = searchParams.get("token") ?? ""

  const [password, setPassword] = useState("")
  const [confirm, setConfirm] = useState("")
  const [error, setError] = useState("")
  const [success, setSuccess] = useState(false)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError("")
    if (password !== confirm) {
      setError(t("reset_password.error_mismatch"))
      return
    }
    if (password.length < 8) {
      setError(t("reset_password.error_too_short"))
      return
    }
    setLoading(true)
    try {
      await authApi.confirmReset(token, password)
      setSuccess(true)
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? t("common.error"))
    } finally {
      setLoading(false)
    }
  }

  if (!token) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-muted/40">
        <div className="w-full max-w-sm p-8 bg-white rounded-xl shadow text-center space-y-4">
          <p className="text-destructive text-sm">{t("reset_password.invalid_link")}</p>
        </div>
      </div>
    )
  }

  if (success) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-muted/40">
        <div className="w-full max-w-sm p-8 bg-white rounded-xl shadow text-center space-y-4">
          <p className="text-green-700 font-medium">{t("reset_password.success")}</p>
          <Button className="w-full" onClick={() => navigate("/login")}>
            {t("reset_password.go_to_login")}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-muted/40">
      <div className="w-full max-w-sm p-8 bg-white rounded-xl shadow space-y-6">
        <div>
          <h1 className="text-lg font-semibold">{t("reset_password.title")}</h1>
          <p className="text-sm text-muted-foreground mt-1">{t("reset_password.subtitle")}</p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1">
            <label className="text-sm font-medium">{t("reset_password.new_password")}</label>
            <Input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoFocus
            />
          </div>
          <div className="space-y-1">
            <label className="text-sm font-medium">{t("reset_password.confirm_password")}</label>
            <Input
              type="password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              required
            />
          </div>
          {error && <p className="text-xs text-destructive">{error}</p>}
          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? t("common.loading") : t("reset_password.submit")}
          </Button>
        </form>
      </div>
    </div>
  )
}
