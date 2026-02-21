import React, { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../lib/api'
import { clearAuth, getToken, getTenantId } from '../lib/auth'

const SERVICES = ['api', 'database', 'redis', 'vector_store', 'llm']
const SERVICE_LABELS = { api: 'API', database: 'Database', redis: 'Redis', vector_store: 'Vector Store', llm: 'LLM' }

function statusColor(status) {
  if (status === true) return 'bg-green-500'
  if (status === false) return 'bg-red-500'
  if (!status) return 'bg-gray-500'
  const s = String(status).toLowerCase()
  if (s === 'ok' || s === 'healthy') return 'bg-green-500'
  if (s === 'degraded') return 'bg-yellow-500'
  return 'bg-red-500'
}

function statusLabel(status) {
  if (status === true) return 'ok'
  if (status === false) return 'error'
  return status || ''
}

export default function Settings() {
  const navigate = useNavigate()
  const [health, setHealth] = useState(null)
  const [healthLoading, setHealthLoading] = useState(false)
  const [copied, setCopied] = useState(false)

  const fetchHealth = useCallback(async () => {
    setHealthLoading(true)
    try {
      const data = await api.get('/health')
      setHealth(data)
    } catch {
      setHealth(null)
    } finally {
      setHealthLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchHealth()
  }, [fetchHealth])

  function maskedToken() {
    const token = getToken()
    if (!token) return 'No token'
    if (token.length <= 12) return token
    return token.slice(0, 8) + '...' + token.slice(-4)
  }

  async function copyToken() {
    const token = getToken()
    if (!token) return
    await navigator.clipboard.writeText(token)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  function handleLogout() {
    clearAuth()
    api.clearToken()
    navigate('/login')
  }

  const tenantId = getTenantId() || 'Unknown'

  return (
    <div className="min-h-screen bg-gray-950 text-white p-8">
      <h1 className="text-3xl font-bold mb-8">Settings</h1>

      <div className="space-y-6 max-w-2xl">
        {/* System Health */}
        <section className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">System Health</h2>
            <button
              onClick={fetchHealth}
              disabled={healthLoading}
              className="px-3 py-1 text-sm bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded transition-colors disabled:opacity-50"
            >
              {healthLoading ? 'Checking...' : 'Refresh'}
            </button>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {SERVICES.map((service) => {
              const status = health?.services?.[service] || (health ? 'unknown' : null)
              return (
                <div key={service} className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${statusColor(status)}`} />
                  <span className="text-sm text-gray-300">{SERVICE_LABELS[service]}</span>
                  <span className={`text-xs ml-auto ${statusColor(status) === 'bg-green-500' ? 'text-green-400' : statusColor(status) === 'bg-red-500' ? 'text-red-400' : 'text-gray-500'}`}>
                    {statusLabel(status)}
                  </span>
                </div>
              )
            })}
          </div>
        </section>

        {/* API Key */}
        <section className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">API Key</h2>
          <div className="flex items-center gap-3">
            <code className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300 font-mono">
              {maskedToken()}
            </code>
            <button
              onClick={copyToken}
              className="px-3 py-2 text-sm bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded transition-colors"
            >
              {copied ? 'Copied' : 'Copy'}
            </button>
            <button
              onClick={handleLogout}
              className="px-3 py-2 text-sm bg-red-600 hover:bg-red-500 rounded transition-colors"
            >
              Clear & Logout
            </button>
          </div>
        </section>

        {/* Tenant */}
        <section className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Tenant</h2>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Tenant ID:</span>
            <code className="text-sm font-mono text-gray-300">{tenantId}</code>
          </div>
        </section>

        {/* Budget */}
        <section className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Budget</h2>
          <p className="text-sm text-gray-500">Cost tracking coming soon</p>
        </section>
      </div>
    </div>
  )
}
