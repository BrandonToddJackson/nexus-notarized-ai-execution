import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { isAuthenticated } from './lib/auth'
import { useAppStore } from './stores/appStore'
import { Sidebar } from './components/layout/Sidebar'
import { TopBar } from './components/layout/TopBar'
import { MobileNav } from './components/layout/MobileNav'

import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Execute from './pages/Execute'
import Ledger from './pages/Ledger'
import Personas from './pages/Personas'
import Workflows from './pages/Workflows'
import WorkflowEditor from './pages/WorkflowEditor'
import Tools from './pages/Tools'
import Knowledge from './pages/Knowledge'
import Settings from './pages/Settings'
import Skills from './pages/Skills'
import SkillEditor from './pages/SkillEditor'
import Credentials from './pages/Credentials'
import MCPServers from './pages/MCPServers'
import Executions from './pages/Executions'
import Chains from './pages/Chains'

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000 } },
})

function ProtectedRoute({ children }) {
  if (!isAuthenticated()) {
    return <Navigate to="/login" replace />
  }
  return children
}

function Layout({ children }) {
  const sidebarCollapsed = useAppStore(s => s.sidebarCollapsed)
  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      <div className={`flex-1 flex flex-col overflow-hidden transition-all ${sidebarCollapsed ? 'md:ml-16' : 'md:ml-60'}`}>
        <TopBar />
        <main className="flex-1 overflow-auto p-6">{children}</main>
      </div>
      <MobileNav />
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Toaster position="top-right" />
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/workflows/:workflowId/edit" element={<ProtectedRoute><WorkflowEditor /></ProtectedRoute>} />
          <Route path="/workflows/new" element={<ProtectedRoute><WorkflowEditor /></ProtectedRoute>} />
          <Route path="/workflows/:workflowId" element={<ProtectedRoute><WorkflowEditor /></ProtectedRoute>} />
          <Route path="/*" element={
            <ProtectedRoute>
              <Layout>
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/execute" element={<Execute />} />
                  <Route path="/workflows" element={<Workflows />} />
                  <Route path="/executions" element={<Executions />} />
                  <Route path="/credentials" element={<Credentials />} />
                  <Route path="/mcp-servers" element={<MCPServers />} />
                  <Route path="/skills" element={<Skills />} />
                  <Route path="/skills/new" element={<SkillEditor />} />
                  <Route path="/skills/:skillId" element={<SkillEditor />} />
                  <Route path="/tools" element={<Tools />} />
                  <Route path="/personas" element={<Personas />} />
                  <Route path="/knowledge" element={<Knowledge />} />
                  <Route path="/ledger" element={<Ledger />} />
                  <Route path="/chains" element={<Chains />} />
                  <Route path="/settings" element={<Settings />} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </Layout>
            </ProtectedRoute>
          } />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
