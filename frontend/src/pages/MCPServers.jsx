import { useState } from 'react'
import { useMCPServers, useAddMCPServer, useRemoveMCPServer, useMCPServerTools, useReconnectMCPServer } from '../hooks/useMCPServers.js'
import { DataTable } from '../components/shared/DataTable.jsx'
import { EmptyState } from '../components/shared/EmptyState.jsx'
import { StatusBadge } from '../components/shared/StatusBadge.jsx'
import { ConfirmDelete } from '../components/shared/ConfirmDelete.jsx'

function ServerTools({ serverId }) {
  const { data: tools, isLoading } = useMCPServerTools(serverId)
  const toolList = Array.isArray(tools) ? tools : tools?.tools || []
  if (isLoading) return <div className="text-sm text-gray-500">Loading tools...</div>
  if (!toolList.length) return <div className="text-sm text-gray-500">No tools registered</div>
  return (
    <div className="space-y-1">
      {toolList.map(t => (
        <div key={t.name} className="text-sm">
          <span className="font-mono text-gray-700">{t.name}</span>
          {t.description && <span className="text-gray-400 ml-2">- {t.description}</span>}
        </div>
      ))}
    </div>
  )
}

export default function MCPServers() {
  const { data: servers, isLoading } = useMCPServers()
  const addServer = useAddMCPServer()
  const removeServer = useRemoveMCPServer()
  const reconnect = useReconnectMCPServer()
  const [showAdd, setShowAdd] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [form, setForm] = useState({ name: '', transport: 'stdio', command: '', args: '', url: '' })
  const serverList = Array.isArray(servers) ? servers : servers?.servers || []

  const columns = [
    { key: 'name', label: 'Name', render: (v) => <span className="font-medium text-gray-900">{v}</span> },
    { key: 'transport', label: 'Transport', render: (v) => <span className="text-xs bg-gray-100 px-2 py-0.5 rounded">{v}</span> },
    { key: 'status', label: 'Status', render: (v) => <StatusBadge status={v || 'unknown'} /> },
    { key: 'tool_count', label: 'Tools', render: (v) => v ?? '-' },
    { key: 'actions', label: '', render: (_, row) => (
      <div className="flex gap-2">
        <button onClick={() => reconnect.mutate(row.id)} className="text-xs text-indigo-600 hover:text-indigo-800">Reconnect</button>
        <button onClick={() => setDeleteTarget(row)} className="text-xs text-red-500 hover:text-red-700">Remove</button>
      </div>
    )},
  ]

  function handleAdd() {
    const data = {
      name: form.name,
      transport: form.transport,
      ...(form.transport === 'stdio'
        ? { command: form.command, args: form.args ? form.args.split(' ') : [] }
        : { url: form.url }),
    }
    addServer.mutate(data, { onSuccess: () => { setShowAdd(false); setForm({ name: '', transport: 'stdio', command: '', args: '', url: '' }) } })
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900">MCP Servers</h1>
        <button
          onClick={() => setShowAdd(true)}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg"
        >
          Add Server
        </button>
      </div>

      {isLoading ? (
        <div className="text-gray-500">Loading...</div>
      ) : serverList.length === 0 ? (
        <EmptyState
          icon="\uD83D\uDD0C"
          title="No MCP servers"
          description="Connect MCP servers to extend tool capabilities"
          actionLabel="Add Server"
          onAction={() => setShowAdd(true)}
        />
      ) : (
        <div className="bg-white border rounded-lg">
          <DataTable
            columns={columns}
            data={serverList}
            renderExpanded={(row) => <ServerTools serverId={row.id} />}
          />
        </div>
      )}

      {showAdd && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setShowAdd(false)}>
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 p-6" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-semibold mb-4">Add MCP Server</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input value={form.name} onChange={e => setForm(f => ({...f, name: e.target.value}))} className="w-full border border-gray-300 rounded px-3 py-2 text-sm" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Transport</label>
                <div className="flex gap-4">
                  {['stdio', 'sse', 'streamable_http'].map(t => (
                    <label key={t} className="flex items-center gap-1 text-sm">
                      <input type="radio" name="transport" checked={form.transport === t} onChange={() => setForm(f => ({...f, transport: t}))} />
                      {t}
                    </label>
                  ))}
                </div>
              </div>
              {form.transport === 'stdio' ? (
                <>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Command</label>
                    <input value={form.command} onChange={e => setForm(f => ({...f, command: e.target.value}))} className="w-full border border-gray-300 rounded px-3 py-2 text-sm" placeholder="npx" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Arguments</label>
                    <input value={form.args} onChange={e => setForm(f => ({...f, args: e.target.value}))} className="w-full border border-gray-300 rounded px-3 py-2 text-sm" placeholder="-y @mcp/server" />
                  </div>
                </>
              ) : (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">URL</label>
                  <input value={form.url} onChange={e => setForm(f => ({...f, url: e.target.value}))} className="w-full border border-gray-300 rounded px-3 py-2 text-sm" placeholder="http://localhost:3000/mcp" />
                </div>
              )}
              <div className="flex justify-end gap-2 pt-2">
                <button onClick={() => setShowAdd(false)} className="px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded">Cancel</button>
                <button onClick={handleAdd} disabled={!form.name} className="px-3 py-2 text-sm bg-indigo-600 text-white rounded hover:bg-indigo-500 disabled:opacity-50">Add</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {deleteTarget && (
        <ConfirmDelete
          title="Remove MCP Server"
          message={`Remove "${deleteTarget.name}" and all its tools?`}
          onConfirm={() => { removeServer.mutate(deleteTarget.id); setDeleteTarget(null) }}
          onCancel={() => setDeleteTarget(null)}
        />
      )}
    </div>
  )
}
