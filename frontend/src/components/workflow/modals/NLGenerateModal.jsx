import { useState } from 'react'
import api from '../../../lib/api'

const EXAMPLE_PROMPTS = [
  'Send daily email digest of new GitHub issues',
  'Review PRs and post Slack summary',
  'Process uploaded CSV and update database',
]

export default function NLGenerateModal({ onGenerated, onClose }) {
  const [description, setDescription] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleGenerate = async () => {
    if (!description.trim()) return
    setLoading(true)
    setError(null)
    try {
      const workflow = await api.post('/v2/workflows/generate', { description: description.trim() })
      onGenerated(workflow)
    } catch (e) {
      setError(e.message || 'Generation failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-full max-w-lg flex flex-col gap-4 p-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">Generate Workflow</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex flex-col gap-2">
          <label className="text-sm text-gray-400">Describe what this workflow should do</label>
          <textarea
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:border-indigo-500 focus:outline-none min-h-[120px] resize-none"
            placeholder="e.g. Every morning, fetch new GitHub issues, summarize them with AI, and send to Slack..."
            value={description}
            onChange={e => setDescription(e.target.value)}
            disabled={loading}
          />
        </div>

        <div className="flex flex-col gap-2">
          <span className="text-xs text-gray-500">Examples:</span>
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_PROMPTS.map(p => (
              <button
                key={p}
                onClick={() => setDescription(p)}
                className="text-xs px-2 py-1 bg-gray-800 border border-gray-700 rounded-full text-gray-300 hover:border-indigo-500 hover:text-indigo-300 transition-colors"
                disabled={loading}
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="bg-red-900/50 border border-red-700 rounded-lg px-3 py-2 text-sm text-red-300">
            {error}
          </div>
        )}

        <div className="flex gap-3 justify-end">
          <button onClick={onClose} className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors" disabled={loading}>
            Cancel
          </button>
          <button
            onClick={handleGenerate}
            disabled={loading || !description.trim()}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {loading ? (
              <>
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
                Generating...
              </>
            ) : 'Generate Workflow'}
          </button>
        </div>
      </div>
    </div>
  )
}
