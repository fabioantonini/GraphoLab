import { Navigate, Outlet } from "react-router-dom"
import { useAuthStore } from "@/store/auth"
import Sidebar from "./Sidebar"

export default function AppLayout() {
  const { isAuthenticated } = useAuthStore()
  if (!isAuthenticated()) return <Navigate to="/login" replace />

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 overflow-auto bg-background">
        <Outlet />
      </main>
    </div>
  )
}
