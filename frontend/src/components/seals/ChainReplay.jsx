import { useState } from 'react'
import { GateBar } from '../gates/GateBar.jsx'
import { StatusBadge } from '../shared/StatusBadge.jsx'
import api from '../../lib/api.js'

export function ChainReplay({ chain }) {
  const [verifyResult, setVerifyResult] = useState(null)
  const [verifying, setVerifying] = useState(false)
  const steps = chain?.steps || chain?.seals || []

  async function handleVerify() {
    setVerifying(true)
    try {
      const result = await api.post(`/v1/ledger/${chain.id || chain.chain_id}/verify`)
      setVerifyResult(result)
    } catch (e) {
      setVerifyResult({ valid: false, error: e.message })
    } finally {
      setVerifying(false)
    }
  }

  function handleExport() {
    const blob = new Blob([JSON.stringify(chain, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `chain-${chain.id || chain.chain_id || 'export'}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button
          onClick={handleVerify}
          disabled={verifying}
          className="px-3 py-1.5 text-sm bg-indigo-600 text-white rounded hover:bg-indigo-500 disabled:opacity-50"
        >
          {verifying ? 'Verifying...' : 'Verify Chain Integrity'}
        </button>
        <button onClick={handleExport} className="px-3 py-1.5 text-sm border border-gray-300 rounded hover:bg-gray-50">
          Export
        </button>
        {verifyResult && (
          <span className={`text-sm ${verifyResult.valid ? 'text-green-600' : 'text-red-600'}`}>
            {verifyResult.valid ? 'Chain verified' : `Verification failed: ${verifyResult.error || 'invalid'}`}
          </span>
        )}
      </div>

      <div className="relative border-l-2 border-gray-200 ml-4 space-y-4">
        {steps.map((step, i) => (
          <div key={step.id || i} className="ml-6 relative">
            <div className="absolute -left-[1.85rem] top-1 w-3 h-3 rounded-full bg-white border-2 border-indigo-400" />
            <div className="bg-white border rounded-lg p-4">
              <div className="flex items-center gap-3 mb-2">
                <span className="text-sm font-medium text-gray-900">{step.name || step.tool_name || `Step ${i + 1}`}</span>
                {step.persona && <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded">{step.persona}</span>}
                {step.status && <StatusBadge status={step.status} />}
              </div>
              {step.intent && <p className="text-xs text-gray-500 mb-2">{step.intent}</p>}
              {step.gates && <GateBar gates={step.gates} compact />}
              {step.seal_hash && (
                <div className="mt-2 text-xs font-mono text-gray-400" title={step.seal_hash}>
                  Seal: {step.seal_hash.slice(0, 16)}...
                </div>
              )}
              {step.output && (
                <pre className="mt-2 text-xs bg-gray-50 p-2 rounded overflow-x-auto max-h-32">
                  {typeof step.output === 'string' ? step.output : JSON.stringify(step.output, null, 2)}
                </pre>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
