import { NavLink } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { FolderOpen, MessageSquare, Users, Microscope, LogOut, Globe } from "lucide-react"
import { cn } from "@/lib/utils"
import { useAuthStore } from "@/store/auth"
import { authApi } from "@/lib/api"
import { Button } from "@/components/ui/button"
import i18n from "@/i18n"

const links = [
  { to: "/projects", label: "nav.projects", icon: FolderOpen },
  { to: "/rag", label: "nav.rag", icon: MessageSquare },
]

const adminLinks = [
  { to: "/admin/users", label: "nav.admin", icon: Users },
]

export default function Sidebar() {
  const { t } = useTranslation()
  const { user, logout } = useAuthStore()

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
      <nav className="flex-1 space-y-0.5">
        {links.map(({ to, label, icon: Icon }) => (
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
