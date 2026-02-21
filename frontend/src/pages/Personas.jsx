import React, { useState, useEffect, useCallback } from 'react'
import api from '../lib/api'

const TIER_COLORS = {
  cold_start: 'bg-gray-700 text-gray-300',
  established: 'bg-blue-900 text-blue-300',
  trusted: 'bg-green-900 text-green-300',
}

const RISK_COLORS = {
  low: 'bg-green-900 text-green-300',
  medium: 'bg-yellow-900 text-yellow-300',
  high: 'bg-red-900 text-red-300',
}

const EMPTY_FORM = {
  name: '',
  description: '',
  allowed_tools: '',
  resource_scopes: '',
  intent_patterns: '',
  max_ttl_seconds: 60,
  risk_tolerance: 'low',
}

export default function Personas() {
  const [personas, setPersonas] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [modalOpen, setModalOpen] = useState(false)
  const [editingId, setEditingId] = useState(null)
  const [form, setForm] = useState(EMPTY_FORM)
  const [saving, setSaving] = useState(false)

  const fetchPersonas = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.get('/personas')
      setPersonas(Array.isArray(data) ? data : data.personas || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchPersonas()
  }, [fetchPersonas])

  const openCreate = () => {
    setEditingId(null)
    setForm(EMPTY_FORM)
    setModalOpen(true)
  }

  const openEdit = (persona) => {
    setEditingId(persona.id)
    setForm({
      name: persona.name || '',
      description: persona.description || '',
      allowed_tools: (persona.allowed_tools || []).join(', '),
      resource_scopes: (persona.resource_scopes || []).join(', '),
      intent_patterns: (persona.intent_patterns || []).join(', '),
      max_ttl_seconds: persona.max_ttl_seconds || 60,
      risk_tolerance: persona.risk_tolerance || 'low',
    })
    setModalOpen(true)
  }

  const closeModal = () => {
    setModalOpen(false)
    setEditingId(null)
    setError(null)
  }

  const handleSave = async () => {
    setSaving(true)
    setError(null)
    const split = (s) => s ? s.split(',').map((x) => x.trim()).filter(Boolean) : []
    const body = {
      name: form.name,
      description: form.description,
      allowed_tools: split(form.allowed_tools),
      resource_scopes: split(form.resource_scopes),
      intent_patterns: split(form.intent_patterns),
      max_ttl_seconds: Number(form.max_ttl_seconds),
      risk_tolerance: form.risk_tolerance,
    }
    try {
      if (editingId) {
        await api.put(`/personas/${editingId}`, body)
      } else {
        await api.post('/personas', body)
      }
      closeModal()
      await fetchPersonas()
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  // Close modal on Escape
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape' && modalOpen) closeModal()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [modalOpen])

  return (
    <div className="min-h-screen bg-gray-950 text-white p-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Personas</h1>
        <button
          onClick={openCreate}
          className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg font-medium transition-colors"
        >
          New Persona
        </button>
      </div>

      {error && !modalOpen && (
        <div className="bg-red-900/50 border border-red-700 rounded-lg p-4 mb-6 text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500" />
          <span className="ml-3 text-gray-400">Loading personas...</span>
        </div>
      ) : personas.length === 0 ? (
        <div className="text-center py-20 text-gray-500">No personas found</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {personas.map((p) => (
            <div key={p.id || p.name} className="bg-gray-900 border border-gray-800 rounded-lg p-5 flex flex-col gap-3">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-bold text-lg">{p.name}</h3>
                  {p.trust_tier && (
                    <span className={`rounded-full px-2 py-0.5 text-xs font-medium mt-1 inline-block ${TIER_COLORS[p.trust_tier] || TIER_COLORS.cold_start}`}>
                      {p.trust_tier}
                    </span>
                  )}
                </div>
                <button
                  onClick={() => openEdit(p)}
                  className="text-gray-400 hover:text-white text-sm transition-colors"
                >
                  Edit
                </button>
              </div>
              {p.description && <p className="text-gray-400 text-sm">{p.description}</p>}
              {(p.allowed_tools || []).length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {p.allowed_tools.map((t) => (
                    <span key={t} className="bg-gray-800 rounded px-2 py-0.5 text-xs text-gray-300">{t}</span>
                  ))}
                </div>
              )}
              <div className="flex items-center gap-3 text-xs mt-auto">
                <span className={`rounded-full px-2 py-0.5 font-medium ${RISK_COLORS[p.risk_tolerance] || RISK_COLORS.low}`}>
                  {p.risk_tolerance}
                </span>
                <span className="text-gray-500">TTL: {p.max_ttl_seconds}s</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Modal */}
      {modalOpen && (
        <div
          className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4"
          onClick={(e) => { if (e.target === e.currentTarget) closeModal() }}
        >
          <div className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-lg max-h-[90vh] overflow-y-auto p-6">
            <h2 className="text-xl font-bold mb-4">{editingId ? 'Edit Persona' : 'New Persona'}</h2>

            {error && (
              <div className="bg-red-900/50 border border-red-700 rounded-lg p-3 mb-4 text-red-300 text-sm">
                {error}
              </div>
            )}

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-1">Name</label>
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  className="w-full bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 px-3 py-2"
                  placeholder="researcher"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Description</label>
                <textarea
                  value={form.description}
                  onChange={(e) => setForm({ ...form, description: e.target.value })}
                  className="w-full bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 px-3 py-2"
                  rows={2}
                  placeholder="Describe persona behavior..."
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Allowed Tools (comma-separated)</label>
                <input
                  type="text"
                  value={form.allowed_tools}
                  onChange={(e) => setForm({ ...form, allowed_tools: e.target.value })}
                  className="w-full bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 px-3 py-2"
                  placeholder="knowledge_search, web_search"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Resource Scopes (comma-separated)</label>
                <input
                  type="text"
                  value={form.resource_scopes}
                  onChange={(e) => setForm({ ...form, resource_scopes: e.target.value })}
                  className="w-full bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 px-3 py-2"
                  placeholder="kb:*, web:*"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Intent Patterns (comma-separated)</label>
                <textarea
                  value={form.intent_patterns}
                  onChange={(e) => setForm({ ...form, intent_patterns: e.target.value })}
                  className="w-full bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 px-3 py-2"
                  rows={2}
                  placeholder="search for information, find data about"
                />
              </div>
              <div className="flex gap-4">
                <div className="flex-1">
                  <label className="block text-sm text-gray-400 mb-1">Max TTL (seconds)</label>
                  <input
                    type="number"
                    value={form.max_ttl_seconds}
                    onChange={(e) => setForm({ ...form, max_ttl_seconds: e.target.value })}
                    className="w-full bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 px-3 py-2"
                    min={1}
                  />
                </div>
                <div className="flex-1">
                  <label className="block text-sm text-gray-400 mb-1">Risk Tolerance</label>
                  <select
                    value={form.risk_tolerance}
                    onChange={(e) => setForm({ ...form, risk_tolerance: e.target.value })}
                    className="w-full bg-gray-800 border border-gray-700 rounded text-white focus:outline-none focus:ring-1 focus:ring-indigo-500 px-3 py-2"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={closeModal}
                className="px-4 py-2 rounded-lg text-gray-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={saving || !form.name}
                className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
