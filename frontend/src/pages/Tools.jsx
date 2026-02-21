import React, { useState, useEffect } from 'react'
import api from '../lib/api'

const RISK_COLORS = {
  low: 'bg-green-900 text-green-300',
  medium: 'bg-yellow-900 text-yellow-300',
  high: 'bg-red-900 text-red-300',
  critical: 'bg-red-900 text-red-200 font-bold',
}

export default function Tools() {
  const [tools, setTools] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchTools = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await api.get('/tools')
        setTools(Array.isArray(data) ? data : data.tools || [])
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    fetchTools()
  }, [])

  return (
    <div className="min-h-screen bg-gray-950 text-white p-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">
          Tools{!loading && ` (${tools.length})`}
        </h1>
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-700 rounded-lg p-4 mb-6 text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500" />
          <span className="ml-3 text-gray-400">Loading tools...</span>
        </div>
      ) : tools.length === 0 ? (
        <div className="text-center py-20 text-gray-500">No tools registered</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {tools.map((tool) => (
            <div key={tool.name} className="bg-gray-900 border border-gray-800 rounded-lg p-5 flex flex-col gap-3">
              <div className="flex items-start justify-between">
                <h3 className="font-bold text-lg font-mono">{tool.name}</h3>
                {tool.requires_approval && (
                  <span className="rounded-full px-2 py-0.5 text-xs font-medium bg-orange-900 text-orange-300">
                    Approval Required
                  </span>
                )}
              </div>
              {tool.description && (
                <p className="text-gray-400 text-sm">{tool.description}</p>
              )}
              <div className="flex items-center gap-3 mt-auto">
                <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${RISK_COLORS[tool.risk_level] || RISK_COLORS.low}`}>
                  {tool.risk_level}
                </span>
                {tool.resource_pattern && (
                  <span className="text-xs text-gray-500">{tool.resource_pattern}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
