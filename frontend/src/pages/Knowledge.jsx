import React, { useState, useEffect } from 'react'
import api from '../lib/api'
import RagSourcePanel from '../components/knowledge/RagSourcePanel'

export default function Knowledge() {
  // Ingest state
  const [source, setSource] = useState('')
  const [namespace, setNamespace] = useState('default')
  const [content, setContent] = useState('')
  const [ingesting, setIngesting] = useState(false)
  const [ingestMsg, setIngestMsg] = useState(null)

  // Namespaces state
  const [namespaces, setNamespaces] = useState([])
  const [nsLoading, setNsLoading] = useState(true)

  // Query state
  const [query, setQuery] = useState('')
  const [queryNs, setQueryNs] = useState('')
  const [results, setResults] = useState(null)
  const [querying, setQuerying] = useState(false)
  const [queryError, setQueryError] = useState(null)
  const [ragPanelOpen, setRagPanelOpen] = useState(false)
  const [lastRagDocId, setLastRagDocId] = useState(null)

  const fetchNamespaces = async () => {
    setNsLoading(true)
    try {
      const data = await api.get('/knowledge/namespaces')
      setNamespaces(Array.isArray(data) ? data : data.namespaces || [])
    } catch {
      setNamespaces([])
    } finally {
      setNsLoading(false)
    }
  }

  useEffect(() => {
    fetchNamespaces()
  }, [])

  const handleIngest = async (e) => {
    e.preventDefault()
    setIngesting(true)
    setIngestMsg(null)
    try {
      await api.post('/knowledge/ingest', {
        content,
        namespace: namespace || 'default',
        source: source || 'manual',
      })
      setIngestMsg({ type: 'success', text: 'Document ingested successfully' })
      setSource('')
      setNamespace('default')
      setContent('')
      fetchNamespaces()
    } catch (err) {
      setIngestMsg({ type: 'error', text: err.message })
    } finally {
      setIngesting(false)
    }
  }

  const handleQuery = async (e) => {
    e.preventDefault()
    if (!query.trim()) return
    setQuerying(true)
    setQueryError(null)
    setResults(null)
    try {
      const params = new URLSearchParams({ query: query.trim() })
      if (queryNs.trim()) params.set('namespace', queryNs.trim())
      const data = await api.get(`/knowledge/query?${params}`)
      setResults(data.results || [])
    } catch (err) {
      setQueryError(err.message)
    } finally {
      setQuerying(false)
    }
  }

  const scoreColor = (score) => {
    if (score > 0.8) return 'bg-green-900 text-green-300'
    if (score > 0.6) return 'bg-yellow-900 text-yellow-300'
    return 'bg-red-900 text-red-300'
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white p-8">
      <h1 className="text-3xl font-bold mb-6">Knowledge Base</h1>

      {/* Section 1: Ingest */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Ingest Document</h2>
        {ingestMsg && (
          <div className={`rounded-lg p-3 mb-4 text-sm ${
            ingestMsg.type === 'success'
              ? 'bg-green-900/50 border border-green-700 text-green-300'
              : 'bg-red-900/50 border border-red-700 text-red-300'
          }`}>
            {ingestMsg.text}
          </div>
        )}
        <form onSubmit={handleIngest} className="space-y-4">
          <div className="flex gap-4">
            <div className="flex-1">
              <input
                type="text"
                value={source}
                onChange={(e) => setSource(e.target.value)}
                placeholder="Document name"
                className="w-full bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 px-3 py-2"
              />
            </div>
            <div className="flex-1">
              <input
                type="text"
                value={namespace}
                onChange={(e) => setNamespace(e.target.value)}
                placeholder="default"
                className="w-full bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 px-3 py-2"
              />
            </div>
          </div>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Paste document content here..."
            rows={6}
            className="w-full bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 px-3 py-2"
          />
          <button
            type="submit"
            disabled={ingesting || !content.trim()}
            className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {ingesting ? 'Ingesting...' : 'Ingest'}
          </button>
        </form>
      </div>

      {/* Section 1b: Multimodal RAG */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 mb-6">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xl font-semibold">Multimodal RAG</h2>
          <button
            onClick={() => setRagPanelOpen(true)}
            className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg font-medium transition-colors"
          >
            Add RAG Source
          </button>
        </div>
        <p className="text-sm text-gray-400 mb-3">
          Ingest PDFs, URLs, or text via RAGAnything for multimodal campaign context.
        </p>
        {lastRagDocId && (
          <div className="bg-green-900/50 border border-green-700 rounded-lg p-3 text-green-300 text-sm">
            Last ingested: <code className="font-mono">{lastRagDocId}</code>
          </div>
        )}
        <RagSourcePanel
          open={ragPanelOpen}
          onClose={() => setRagPanelOpen(false)}
          onSuccess={(docId) => { setLastRagDocId(docId); setRagPanelOpen(false) }}
        />
      </div>

      {/* Section 2: Namespaces */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Namespaces</h2>
        {nsLoading ? (
          <div className="flex items-center gap-2 text-gray-400">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-500" />
            Loading...
          </div>
        ) : namespaces.length === 0 ? (
          <p className="text-gray-500">No namespaces found</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {namespaces.map((ns) => (
              <button
                key={ns}
                onClick={() => setQueryNs(ns)}
                className="bg-gray-800 px-3 py-1 rounded-full text-sm text-gray-300 hover:bg-gray-700 transition-colors cursor-pointer"
              >
                {ns}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Section 3: Query */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Query Knowledge</h2>
        <form onSubmit={handleQuery} className="flex gap-4 mb-4">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search query..."
            className="flex-1 bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 px-3 py-2"
          />
          <input
            type="text"
            value={queryNs}
            onChange={(e) => setQueryNs(e.target.value)}
            placeholder="Namespace"
            className="w-40 bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 px-3 py-2"
          />
          <button
            type="submit"
            disabled={querying || !query.trim()}
            className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {querying ? 'Searching...' : 'Search'}
          </button>
        </form>

        {queryError && (
          <div className="bg-red-900/50 border border-red-700 rounded-lg p-3 mb-4 text-red-300 text-sm">
            {queryError}
          </div>
        )}

        {querying && (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-500" />
            <span className="ml-3 text-gray-400">Searching...</span>
          </div>
        )}

        {results !== null && !querying && (
          results.length === 0 ? (
            <p className="text-gray-500 text-center py-6">No results</p>
          ) : (
            <div className="space-y-3">
              {results.map((r, i) => (
                <div key={i} className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${scoreColor(r.score)}`}>
                      {r.score != null ? Number(r.score).toFixed(2) : '—'}
                    </span>
                    {(r.metadata?.source || r.source) && (
                      <span className="text-xs text-gray-500">{r.metadata?.source || r.source}</span>
                    )}
                  </div>
                  <p className="text-sm font-mono text-gray-300 whitespace-pre-wrap">
                    {typeof r.content === 'string'
                      ? r.content.slice(0, 200) + (r.content.length > 200 ? '...' : '')
                      : typeof r === 'string'
                        ? r.slice(0, 200) + (r.length > 200 ? '...' : '')
                        : JSON.stringify(r).slice(0, 200)}
                  </p>
                </div>
              ))}
            </div>
          )
        )}
      </div>
    </div>
  )
}
