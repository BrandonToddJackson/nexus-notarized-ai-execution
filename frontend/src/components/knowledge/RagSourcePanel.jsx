import React, { useState } from 'react'

const TABS = [
  { id: 'text', label: 'Text / Notion' },
  { id: 'url', label: 'URL' },
  { id: 'file', label: 'File Upload' },
]

export default function RagSourcePanel({ open, onClose, onSuccess }) {
  const [tab, setTab] = useState('text')
  const [namespace, setNamespace] = useState('campaign')
  const [content, setContent] = useState('')
  const [url, setUrl] = useState('')
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  if (!open) return null

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('namespace', namespace)
      if (tab === 'text') formData.append('content', content)
      else if (tab === 'url') formData.append('url', url)
      else if (tab === 'file' && file) formData.append('file', file)

      // Use fetch directly for FormData (multipart)
      const token = localStorage.getItem('nexus_token') || ''
      const resp = await fetch('/v1/knowledge/multimodal', {
        method: 'POST',
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        body: formData,
      })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(err.detail || `HTTP ${resp.status}`)
      }
      const data = await resp.json()
      onSuccess(data.document_id)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const inputCls = 'w-full bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 px-3 py-2'

  return (
    <div
      className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 max-w-lg w-full">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Add RAG Source</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {/* Namespace */}
        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-1">Namespace</label>
          <input
            type="text"
            value={namespace}
            onChange={(e) => setNamespace(e.target.value)}
            placeholder="campaign"
            className={inputCls}
            aria-label="namespace"
          />
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mb-4 bg-gray-800 rounded-lg p-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`flex-1 py-1.5 rounded-md text-sm font-medium transition-colors ${
                tab === t.id
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {tab === 'text' && (
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Paste Notion content, campaign brief, ICP description..."
              rows={6}
              className={inputCls}
              aria-label="content"
            />
          )}
          {tab === 'url' && (
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/campaign-brief"
              className={inputCls}
              aria-label="url"
            />
          )}
          {tab === 'file' && (
            <div>
              <input
                type="file"
                accept=".pdf,.docx,.txt,.md,.html"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="w-full text-gray-300 text-sm"
                aria-label="file"
              />
              {file && (
                <p className="text-xs text-gray-500 mt-1">{file.name}</p>
              )}
            </div>
          )}

          {error && (
            <div className="bg-red-900/50 border border-red-700 rounded-lg p-3 text-red-300 text-sm">
              {error}
            </div>
          )}

          <div className="flex gap-3 justify-end">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-gray-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || (tab === 'text' && !content.trim()) || (tab === 'url' && !url.trim()) || (tab === 'file' && !file)}
              className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50"
            >
              {loading ? 'Ingesting...' : 'Ingest'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
