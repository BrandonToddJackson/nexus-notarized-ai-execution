import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../lib/api'
import { GenerateModal } from '../components/workflows/GenerateModal'
import { TemplateGallery } from '../components/workflows/TemplateGallery'
import { WorkflowCard } from '../components/workflows/WorkflowCard'

const VIEW_KEY = 'nexus_workflows_view'

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
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [view, setView] = useState(() => localStorage.getItem(VIEW_KEY) || 'card')
  const [showGenerate, setShowGenerate] = useState(false)
  const [showTemplates, setShowTemplates] = useState(false)
  const importRef = useRef(null)

  const load = () => {
    setLoading(true)
    api.get('/v2/workflows')
      .then((data) => setWorkflows(Array.isArray(data) ? data : data.workflows || []))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const setViewPersisted = (v) => {
    setView(v)
    localStorage.setItem(VIEW_KEY, v)
  }

  const handleImport = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      const text = await file.text()
      await api.post('/v2/workflows/import', { data: text })
      load()
    } catch (err) {
      setError(err.message)
    } finally {
      importRef.current.value = ''
    }
  }

  const handleGenerated = (workflow) => {
    setShowGenerate(false)
    if (workflow?.id) navigate(`/workflows/${workflow.id}`)
    else load()
  }

  const handleTemplateCreate = () => {
    setShowTemplates(false)
    load()
  }

  const filtered = workflows.filter((wf) => {
    if (statusFilter !== 'all' && wf.status !== statusFilter) return false
    if (search) {
      const q = search.toLowerCase()
      return (wf.name || '').toLowerCase().includes(q) || (wf.description || '').toLowerCase().includes(q)
    }
    return true
  })

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Workflows</h1>
          <p className="text-sm text-gray-400 mt-1">
            Visual automation workflows with NEXUS security gates
          </p>
        </div>
        {/* Toolbar */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowTemplates(true)}
            className="px-3 py-2 text-sm text-gray-300 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg transition-colors"
          >
            Templates
          </button>
          <button
            onClick={() => importRef.current?.click()}
            className="px-3 py-2 text-sm text-gray-300 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg transition-colors"
          >
            Import
          </button>
          <button
            onClick={() => setShowGenerate(true)}
            className="px-3 py-2 text-sm text-indigo-300 bg-indigo-900/40 hover:bg-indigo-900/70 border border-indigo-700 rounded-lg transition-colors"
          >
            ✨ Generate with AI
          </button>
          <button
            onClick={() => navigate('/workflows/new')}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors"
          >
            + New Workflow
          </button>
        </div>
      </div>

      {/* Filters + View toggle */}
      <div className="flex items-center gap-3 mb-5">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search workflows…"
          className="flex-1 max-w-xs px-3 py-1.5 text-sm bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500"
        />
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="px-3 py-1.5 text-sm bg-gray-900 border border-gray-700 rounded-lg text-gray-300 focus:outline-none"
        >
          <option value="all">All statuses</option>
          <option value="active">Active</option>
          <option value="draft">Draft</option>
          <option value="paused">Paused</option>
          <option value="archived">Archived</option>
        </select>
        <div className="flex rounded-lg overflow-hidden border border-gray-700 ml-auto">
          <button
            onClick={() => setViewPersisted('card')}
            className={`px-3 py-1.5 text-sm transition-colors ${view === 'card' ? 'bg-indigo-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
          >⊞</button>
          <button
            onClick={() => setViewPersisted('table')}
            className={`px-3 py-1.5 text-sm transition-colors ${view === 'table' ? 'bg-indigo-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
          >☰</button>
        </div>
      </div>

      {/* Hidden import input */}
      <input ref={importRef} type="file" accept=".json" className="hidden" onChange={handleImport} />

      {/* Loading */}
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

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg p-4 text-sm text-red-300 mb-4">
          {error}
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && filtered.length === 0 && (
        <div className="text-center py-16">
          <div className="text-4xl mb-4">🔄</div>
          <h3 className="text-lg font-medium text-gray-300 mb-2">
            {workflows.length === 0 ? 'No workflows yet' : 'No workflows match your filters'}
          </h3>
          <p className="text-sm text-gray-500 mb-4">
            {workflows.length === 0 ? 'Create your first workflow to get started.' : 'Try a different search or status filter.'}
          </p>
          {workflows.length === 0 && (
            <button
              onClick={() => navigate('/workflows/new')}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors"
            >
              Create Workflow
            </button>
          )}
        </div>
      )}

      {/* Card view */}
      {!loading && !error && filtered.length > 0 && view === 'card' && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((wf) => (
            <WorkflowCard key={wf.id} workflow={wf} onRefresh={load} />
          ))}
        </div>
      )}

      {/* Table view */}
      {!loading && !error && filtered.length > 0 && view === 'table' && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="border-b border-gray-800">
              <tr className="text-left text-gray-500 text-xs uppercase tracking-wide">
                <th className="px-4 py-3">Name</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Steps</th>
                <th className="px-4 py-3">Updated</th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {filtered.map((wf) => (
                <tr key={wf.id} className="hover:bg-gray-800/50 transition-colors">
                  <td className="px-4 py-3 text-white font-medium">{wf.name || 'Untitled'}</td>
                  <td className="px-4 py-3">
                    <span className={`text-xs px-2 py-0.5 rounded-full border ${STATUS_BADGE[wf.status] || STATUS_BADGE.draft}`}>
                      {wf.status || 'draft'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-400">{wf.steps?.length ?? 0}</td>
                  <td className="px-4 py-3 text-gray-500">{formatDate(wf.updated_at)}</td>
                  <td className="px-4 py-3">
                    <button
                      onClick={() => navigate(`/workflows/${wf.id}`)}
                      className="text-indigo-400 hover:text-indigo-300 text-xs"
                    >
                      Edit →
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Modals */}
      {showGenerate && (
        <GenerateModal onGenerated={handleGenerated} onClose={() => setShowGenerate(false)} />
      )}
      {showTemplates && (
        <TemplateGallery onCreated={handleTemplateCreate} onClose={() => setShowTemplates(false)} />
      )}
    </div>
  )
}
