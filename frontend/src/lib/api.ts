/**
 * GraphoLab API client.
 * All requests go through /api (proxied to http://localhost:8000 in dev).
 * Access token is attached automatically; on 401 a refresh is attempted once.
 */
import axios, { AxiosError, type InternalAxiosRequestConfig } from "axios"
import { useAuthStore } from "@/store/auth"

export const api = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
})

// ── Request interceptor: attach access token ──────────────────────────────────
api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = useAuthStore.getState().accessToken
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// ── Response interceptor: auto-refresh on 401 ────────────────────────────────
let _refreshing = false
let _queue: Array<(token: string) => void> = []

api.interceptors.response.use(
  (res) => res,
  async (error: AxiosError) => {
    const original = error.config as InternalAxiosRequestConfig & { _retry?: boolean }
    if (error.response?.status !== 401 || original._retry) {
      return Promise.reject(error)
    }
    original._retry = true

    if (_refreshing) {
      return new Promise((resolve) => {
        _queue.push((token) => {
          original.headers.Authorization = `Bearer ${token}`
          resolve(api(original))
        })
      })
    }

    _refreshing = true
    try {
      const { refreshToken, setTokens, logout } = useAuthStore.getState()
      if (!refreshToken) { logout(); return Promise.reject(error) }

      const res = await axios.post("/api/auth/refresh", { refresh_token: refreshToken })
      const { access_token, refresh_token } = res.data
      setTokens(access_token, refresh_token)

      _queue.forEach((cb) => cb(access_token))
      _queue = []
      original.headers.Authorization = `Bearer ${access_token}`
      return api(original)
    } catch {
      useAuthStore.getState().logout()
      return Promise.reject(error)
    } finally {
      _refreshing = false
    }
  }
)

// ── Typed helpers ─────────────────────────────────────────────────────────────

export interface TokenResponse {
  access_token: string
  refresh_token: string
}

export interface User {
  id: number
  email: string
  full_name: string
  role: "admin" | "examiner" | "viewer"
  is_active: boolean
  organization_id: number | null
}

export interface Project {
  id: number
  title: string
  description: string | null
  status: "draft" | "in_progress" | "completed" | "archived"
  owner_id: number
  document_count: number
}

export interface Document {
  id: number
  filename: string
  content_type: string
  size_bytes: number
  storage_key: string
}

export interface Analysis {
  id: number
  analysis_type: string
  result_text: string | null
  project_id: number
  document_id: number | null
}

// Auth
export const authApi = {
  login: (email: string, password: string) =>
    api.post<TokenResponse>("/auth/login", new URLSearchParams({ username: email, password }), {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    }),
  refresh: (refreshToken: string) =>
    api.post<TokenResponse>("/auth/refresh", { refresh_token: refreshToken }),
  logout: () => api.post("/auth/logout"),
}

// Users
export const usersApi = {
  me: () => api.get<User>("/users/me"),
  list: () => api.get<User[]>("/users/"),
  create: (data: { email: string; full_name: string; password: string; role?: string }) =>
    api.post<User>("/users/", data),
  updateMe: (data: { full_name?: string; current_password?: string; new_password?: string }) =>
    api.put<User>("/users/me", data),
  deactivate: (id: number) => api.delete(`/users/${id}`),
}

// Projects
export const projectsApi = {
  list: () => api.get<Project[]>("/projects/"),
  create: (data: { title: string; description?: string }) => api.post<Project>("/projects/", data),
  get: (id: number) => api.get<Project>(`/projects/${id}`),
  update: (id: number, data: Partial<Project>) => api.put<Project>(`/projects/${id}`, data),
  delete: (id: number) => api.delete(`/projects/${id}`),
  uploadDocument: (projectId: number, file: File) => {
    const fd = new FormData()
    fd.append("file", file)
    return api.post<Document>(`/projects/${projectId}/documents`, fd, {
      headers: { "Content-Type": "multipart/form-data" },
    })
  },
  listDocuments: (projectId: number) => api.get<Document[]>(`/projects/${projectId}/documents`),
  deleteDocument: (projectId: number, docId: number) =>
    api.delete(`/projects/${projectId}/documents/${docId}`),
}

// Analysis
export const analysisApi = {
  run: (type: string, projectId: number, documentId: number) => {
    const pathMap: Record<string, string> = { writer_identification: "writer" }
    const path = pathMap[type] ?? type.replaceAll("_", "-")
    return api.post<Analysis>(`/analysis/${path}`, { project_id: projectId, document_id: documentId })
  },
  runSignatureVerification: (projectId: number, documentId: number, referenceDocumentId: number) =>
    api.post<Analysis>("/analysis/signature-verification", { project_id: projectId, document_id: documentId, reference_document_id: referenceDocumentId }),
  runPipeline: (projectId: number, documentId: number) =>
    api.post<Analysis>("/analysis/pipeline", { project_id: projectId, document_id: documentId }),
  list: (projectId: number) => api.get<Analysis[]>(`/analysis/project/${projectId}`),
}

// RAG
export const ragApi = {
  status: () => api.get<{ ollama_reachable: boolean; models: string[] }>("/rag/status"),
  listDocs: () => api.get<{ filename: string; chunks: number }[]>("/rag/docs"),
  addDoc: (file: File) => {
    const fd = new FormData()
    fd.append("file", file)
    return api.post("/rag/docs", fd, { headers: { "Content-Type": "multipart/form-data" } })
  },
  removeDoc: (filename: string) => api.delete(`/rag/docs/${encodeURIComponent(filename)}`),
}
