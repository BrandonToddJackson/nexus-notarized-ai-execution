import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../lib/api'

const STATUS_BADGE = {
  active: 'bg-green-900/50 text-green-400 border-green-800',
  paused: 'bg-yellow-900/50 text-yellow-400 border-yellow-800',
  draft: 'bg-gray-800 text-gray-400 border-gray-700',
  archived: 'bg-gray-800 text-gray-500 border-gray-700',
}

function formatDate(d) {
  if (!d) return ''
  const date = new Date(d)
  const now = new Date()
  const diff = now - date
  if (diff < 60000) return 'just now'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
  return date.toLocaleDateString()
}

export default function Workflows() {
  const navigate = useNavigate()
  const [workflows, setWorkflows] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.get('/v2/workflows')
      .then((data) => setWorkflows(Array.isArray(data) ? data : data.workflows || []))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">Workflows</h1>
          <p className="text-sm text-gray-400 mt-1">
            Visual automation workflows with NEXUS security gates
          </p>
        </div>
        <button
          onClick={() => navigate('/workflows/new')}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors"
        >
          + New Workflow
        </button>
      </div>

      {loading && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-gray-900 border border-gray-800 rounded-lg p-4 animate-pulse">
              <div className="h-4 bg-gray-800 rounded w-2/3 mb-3" />
              <div className="h-3 bg-gray-800 rounded w-1/3" />
            </div>
          ))}
        </div>
      )}

      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg p-4 text-sm text-red-300">
          {error}
        </div>
      )}

      {!loading && !error && workflows.length === 0 && (
        <div className="text-center py-16">
          <svg className="w-16 h-16 text-gray-700 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
          <h3 className="text-lg font-medium text-gray-300 mb-2">No workflows yet</h3>
          <p className="text-sm text-gray-500 mb-4">Create your first workflow to get started.</p>
          <button
            onClick={() => navigate('/workflows/new')}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors"
          >
            Create Workflow
          </button>
        </div>
      )}

      {!loading && !error && workflows.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {workflows.map((wf) => (
            <div
              key={wf.id}
              className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors"
            >
              <div className="flex items-start justify-between mb-3">
                <h3 className="text-sm font-semibold text-white truncate pr-2">
                  {wf.name || 'Untitled'}
                </h3>
                <span
                  className={`text-xs px-2 py-0.5 rounded-full border shrink-0 ${
                    STATUS_BADGE[wf.status] || STATUS_BADGE.draft
                  }`}
                >
                  {wf.status || 'draft'}
                </span>
              </div>
              <div className="flex items-center gap-3 text-xs text-gray-500">
                <span>v{wf.version || 1}</span>
                {wf.updated_at && <span>{formatDate(wf.updated_at)}</span>}
              </div>
              <button
                onClick={() => navigate(`/workflows/${wf.id}`)}
                className="mt-3 w-full py-1.5 text-xs text-indigo-300 bg-indigo-900/30 hover:bg-indigo-900/60 rounded transition-colors"
              >
                Edit
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
