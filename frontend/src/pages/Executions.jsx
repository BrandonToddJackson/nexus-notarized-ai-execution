import { useState } from 'react'
import { useExecutions } from '../hooks/useExecutions.js'
import { useExecutionStream } from '../hooks/useExecutionStream.js'
import { DataTable } from '../components/shared/DataTable.jsx'
import { StatusBadge } from '../components/shared/StatusBadge.jsx'
import { EmptyState } from '../components/shared/EmptyState.jsx'
import { SearchInput } from '../components/shared/SearchInput.jsx'
import { ChainReplay } from '../components/seals/ChainReplay.jsx'

function CopyableId({ id }) {
  const [copied, setCopied] = useState(false)
  function handleCopy() {
    navigator.clipboard.writeText(id)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }
  return (
    <button onClick={handleCopy} className="font-mono text-xs text-gray-600 hover:text-gray-900" title={id}>
      {id.slice(0, 8)}... {copied ? '(copied)' : ''}
    </button>
  )
}

export default function Executions() {
  useExecutionStream()
  const [status, setStatus] = useState('')
  const [gateFailures, setGateFailures] = useState(false)
  const { data: executions, isLoading } = useExecutions({
    status: status || undefined,
    gate_failures: gateFailures || undefined,
  })
  const execList = Array.isArray(executions) ? executions : executions?.executions || []

  const columns = [
    { key: 'id', label: 'ID', render: (v) => <CopyableId id={v} /> },
    { key: 'workflow_name', label: 'Workflow', render: (v) => v || '-' },
    { key: 'status', label: 'Status', render: (v) => <StatusBadge status={v} /> },
    { key: 'duration_ms', label: 'Duration', render: (v) => v != null ? `${(v / 1000).toFixed(1)}s` : '-' },
    { key: 'step_count', label: 'Steps', render: (v) => v ?? '-' },
    { key: 'gate_failure_count', label: 'Gate Failures', render: (v) =>
      v > 0 ? <span className="text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded-full">{v}</span> : '-'
    },
    { key: 'started_at', label: 'Started', render: (v) => v ? new Date(v).toLocaleString() : '-' },
  ]

  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Executions</h1>
      <div className="flex gap-4 mb-4 items-center">
        <select
          value={status}
          onChange={e => setStatus(e.target.value)}
          className="border border-gray-300 rounded px-3 py-2 text-sm"
        >
          <option value="">All statuses</option>
          <option value="running">Running</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
          <option value="blocked">Blocked</option>
        </select>
        <label className="flex items-center gap-2 text-sm text-gray-700">
          <input type="checkbox" checked={gateFailures} onChange={e => setGateFailures(e.target.checked)} />
          Has gate failures
        </label>
      </div>

      {isLoading ? (
        <div className="text-gray-500">Loading...</div>
      ) : execList.length === 0 ? (
        <EmptyState
          icon="\uD83D\uDCCB"
          title="No executions"
          description="Execute a workflow or task to see results here"
        />
      ) : (
        <div className="bg-white border rounded-lg">
          <DataTable
            columns={columns}
            data={execList}
            renderExpanded={(row) => <ChainReplay chain={row} />}
          />
        </div>
      )}
    </div>
  )
}
