import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'

// Pages — each needs full implementation
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Execute from './pages/Execute'
import Ledger from './pages/Ledger'
import Personas from './pages/Personas'
import Tools from './pages/Tools'
import Knowledge from './pages/Knowledge'
import Settings from './pages/Settings'

// TODO: Add auth guard, layout wrapper, error boundary

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/" element={<Dashboard />} />
        <Route path="/execute" element={<Execute />} />
        <Route path="/ledger" element={<Ledger />} />
        <Route path="/personas" element={<Personas />} />
        <Route path="/tools" element={<Tools />} />
        <Route path="/knowledge" element={<Knowledge />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
