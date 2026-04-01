import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import AppLayout from "@/components/layout/AppLayout"
import LoginPage from "@/pages/LoginPage"
import ProjectsPage from "@/pages/ProjectsPage"
import ProjectDetailPage from "@/pages/ProjectDetailPage"
import RagPage from "@/pages/RagPage"
import AdminPage from "@/pages/AdminPage"

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route element={<AppLayout />}>
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/projects/:id" element={<ProjectDetailPage />} />
          <Route path="/rag" element={<RagPage />} />
          <Route path="/admin/*" element={<AdminPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/projects" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
