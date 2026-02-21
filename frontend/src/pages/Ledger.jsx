import React, { useState, useEffect, useCallback } from 'react'
import api from '../lib/api'

const PAGE_SIZE = 50

const STATUS_COLORS = {
  executed: 'bg-green-900 text-green-300',
  blocked: 'bg-red-900 text-red-300',
  failed: 'bg-orange-900 text-orange-300',
  pending: 'bg-gray-700 text-gray-300',
}

export default function Ledger() {
  const [seals, setSeals] = useState([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(true)
  const [expandedId, setExpandedId] = useState(null)
  const [error, setError] = useState(null)

  const fetchSeals = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.get(`/ledger?limit=${PAGE_SIZE}&offset=${offset}`)
      // Handle both {seals: [...], total} and raw array responses
      if (Array.isArray(data)) {
        setSeals(data)
        setTotal(data.length)
      } else {
        setSeals(data.seals || [])
        setTotal(data.total ?? (data.seals || []).length)
      }
    } catch (err) {
      setError(err.message)
      setSeals([])
    } finally {
      setLoading(false)
    }
  }, [offset])

  useEffect(() => {
    fetchSeals()
  }, [fetchSeals])

  const short = (id) => id ? String(id).slice(0, 8) : '—'
  const currentPage = Math.floor(offset / PAGE_SIZE)
  const totalPages = Math.ceil(total / PAGE_SIZE)
  const showStart = total === 0 ? 0 : offset + 1
  const showEnd = Math.min(offset + PAGE_SIZE, total)

  return (
    <div className="min-h-screen bg-gray-950 text-white p-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Ledger</h1>
        <button
          onClick={fetchSeals}
          disabled={loading}
          className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50"
        >
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-700 rounded-lg p-4 mb-6 text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500" />
          <span className="ml-3 text-gray-400">Loading seals...</span>
        </div>
      ) : seals.length === 0 ? (
        <div className="text-center py-20 text-gray-500">No seals found</div>
      ) : (
        <>
          <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800 text-gray-400 text-left">
                  <th className="px-4 py-3">Seal ID</th>
                  <th className="px-4 py-3">Chain ID</th>
                  <th className="px-4 py-3">Step</th>
                  <th className="px-4 py-3">Persona</th>
                  <th className="px-4 py-3">Tool</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Created At</th>
                </tr>
              </thead>
              <tbody>
                {seals.map((seal) => {
                  const sealId = seal.id || seal.seal_id
                  const isExpanded = expandedId === sealId
                  return (
                    <React.Fragment key={sealId}>
                      <tr
                        className="border-b border-gray-800 hover:bg-gray-800/50 cursor-pointer transition-colors"
                        onClick={() => setExpandedId(isExpanded ? null : sealId)}
                      >
                        <td className="px-4 py-3 font-mono text-xs">{short(sealId)}</td>
                        <td className="px-4 py-3 font-mono text-xs">{short(seal.chain_id)}</td>
                        <td className="px-4 py-3">{seal.step_index ?? '—'}</td>
                        <td className="px-4 py-3">{seal.persona || seal.persona_name || '—'}</td>
                        <td className="px-4 py-3">{seal.tool || seal.tool_name || '—'}</td>
                        <td className="px-4 py-3">
                          <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_COLORS[seal.status] || STATUS_COLORS.pending}`}>
                            {seal.status}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-gray-400 text-xs">
                          {seal.created_at ? new Date(seal.created_at).toLocaleString() : '—'}
                        </td>
                      </tr>
                      {isExpanded && (
                        <tr>
                          <td colSpan={7} className="px-4 py-4 bg-gray-800/30">
                            <ExpandedSealDetails seal={seal} />
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between mt-4">
            <span className="text-sm text-gray-400">
              Showing {showStart}–{showEnd} of {total}
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                disabled={offset === 0}
                className="bg-gray-800 hover:bg-gray-700 text-white px-3 py-1.5 rounded text-sm disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                Prev
              </button>
              <span className="text-sm text-gray-500 flex items-center px-2">
                Page {currentPage + 1} of {Math.max(totalPages, 1)}
              </span>
              <button
                onClick={() => setOffset(offset + PAGE_SIZE)}
                disabled={offset + PAGE_SIZE >= total}
                className="bg-gray-800 hover:bg-gray-700 text-white px-3 py-1.5 rounded text-sm disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function ExpandedSealDetails({ seal }) {
  const gates = seal.gates || []
  const reasoning = seal.reasoning || seal.cot_reasoning || []

  return (
    <div className="space-y-4">
      {/* Gate details */}
      {gates.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-gray-300 mb-2">Gate Results</h4>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 text-left text-xs">
                <th className="pr-4 pb-1">Gate Name</th>
                <th className="pr-4 pb-1">Verdict</th>
                <th className="pr-4 pb-1">Score</th>
                <th className="pr-4 pb-1">Threshold</th>
              </tr>
            </thead>
            <tbody>
              {gates.map((g, i) => (
                <tr key={i} className="text-gray-300">
                  <td className="pr-4 py-1">{g.name || g.gate_name}</td>
                  <td className="pr-4 py-1">
                    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                      g.verdict === 'pass' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                    }`}>
                      {g.verdict}
                    </span>
                  </td>
                  <td className="pr-4 py-1 font-mono">{g.score != null ? Number(g.score).toFixed(3) : '—'}</td>
                  <td className="pr-4 py-1 font-mono">{g.threshold != null ? Number(g.threshold).toFixed(3) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* CoT reasoning */}
      {reasoning.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-gray-300 mb-2">Chain of Thought</h4>
          <ol className="list-decimal list-inside space-y-1 font-mono text-xs text-gray-400 bg-gray-900 rounded p-3">
            {reasoning.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ol>
        </div>
      )}

      {gates.length === 0 && reasoning.length === 0 && (
        <p className="text-sm text-gray-500">No gate or reasoning data available for this seal.</p>
      )}
    </div>
  )
}
