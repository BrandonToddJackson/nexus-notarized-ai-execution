import React, { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../lib/api'
import LiveFeed from '../components/LiveFeed'

export default function Dashboard() {
  const navigate = useNavigate()
  const [seals, setSeals] = useState([])
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [error, setError] = useState(null)

  const fetchData = useCallback(async () => {
    try {
      const data = await api.get('/ledger?limit=20')
      const sealList = Array.isArray(data) ? data : (data.seals || data.items || [])
      setSeals(sealList)
      setEvents(sealList.map(seal => ({
        type: 'seal_created',
        data: seal,
        timestamp: new Date(seal.created_at || Date.now()),
      })))
      setLastUpdated(new Date())
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 10000)
    return () => clearInterval(interval)
  }, [fetchData])

  // Compute stats
  const chainIds = [...new Set(seals.map(s => s.chain_id).filter(Boolean))]
  const totalChains = chainIds.length
  const totalSeals = seals.length
  const blockedCount = seals.filter(s => s.status === 'blocked').length
  const blockedPct = totalSeals > 0 ? ((blockedCount / totalSeals) * 100).toFixed(1) : '0.0'
  const totalCost = seals.reduce((sum, s) => sum + (s.cost_usd || 0), 0)

  // Recent chains: last 5 unique chain_ids
  const recentChains = chainIds.slice(0, 5).map(chainId => {
    const chainSeals = seals.filter(s => s.chain_id === chainId)
    const hasBlocked = chainSeals.some(s => s.status === 'blocked')
    return {
      chainId,
      sealCount: chainSeals.length,
      hasBlocked,
    }
  })

  const stats = [
    { label: 'Total Chains', value: totalChains, color: 'text-white' },
    { label: 'Total Seals', value: totalSeals, color: 'text-white' },
    { label: 'Blocked %', value: `${blockedPct}%`, color: parseFloat(blockedPct) > 10 ? 'text-red-400' : 'text-white' },
    { label: 'LLM Cost', value: `$${totalCost.toFixed(2)}`, color: 'text-white' },
  ]

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <div className="p-6 border-b border-gray-800 flex items-center justify-between">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        {lastUpdated && (
          <span className="text-xs text-gray-500">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </span>
        )}
      </div>

      <div className="p-6 space-y-6">
        {/* Stat cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {stats.map(stat => (
            <div key={stat.label} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <div className={`text-3xl font-bold ${stat.color}`}>
                {loading ? '-' : stat.value}
              </div>
              <div className="text-sm text-gray-400 mt-1">{stat.label}</div>
            </div>
          ))}
        </div>

        {error && (
          <div className="bg-red-900/30 border border-red-800 rounded-lg p-3 text-sm text-red-300">
            Failed to load data: {error}
          </div>
        )}

        {/* Middle section */}
        <div className="flex gap-6">
          {/* Live feed */}
          <div className="flex-1 min-w-0">
            <h2 className="text-sm font-medium text-gray-300 mb-3">Live Feed</h2>
            <LiveFeed events={events} />
          </div>

          {/* Recent chains */}
          <div className="w-80 flex-shrink-0">
            <h2 className="text-sm font-medium text-gray-300 mb-3">Recent Chains</h2>
            <div className="space-y-3">
              {recentChains.length === 0 && !loading && (
                <p className="text-xs text-gray-500">No chains recorded yet.</p>
              )}
              {loading && recentChains.length === 0 && (
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 animate-pulse">
                  <div className="h-4 bg-gray-800 rounded w-24 mb-2" />
                  <div className="h-3 bg-gray-800 rounded w-16" />
                </div>
              )}
              {recentChains.map(chain => (
                <button
                  key={chain.chainId}
                  onClick={() => navigate('/ledger')}
                  className="w-full text-left bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-600 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sm text-gray-200">
                      {chain.chainId.slice(0, 8)}
                    </span>
                    <span
                      className={`text-xs font-medium rounded-full px-2 py-0.5 ${
                        chain.hasBlocked
                          ? 'bg-red-900 text-red-300'
                          : 'bg-green-900 text-green-300'
                      }`}
                    >
                      {chain.hasBlocked ? 'blocked' : 'executed'}
                    </span>
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {chain.sealCount} seal{chain.sealCount !== 1 ? 's' : ''}
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
