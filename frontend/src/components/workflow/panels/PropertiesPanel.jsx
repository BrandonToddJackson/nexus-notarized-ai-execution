import { useState, useEffect } from 'react'
import api from '../../../lib/api'

export default function PropertiesPanel({ selectedNode, selectedEdge, onChange }) {
  const [tools, setTools] = useState([])
  const [personas, setPersonas] = useState([])

  useEffect(() => {
    api.get('/tools').then(data => setTools(Array.isArray(data) ? data : (data.tools || []))).catch(() => {})
    api.get('/personas').then(data => setPersonas(Array.isArray(data) ? data : (data.personas || []))).catch(() => {})
  }, [])

  if (!selectedNode && !selectedEdge) {
    return (
      <div className="w-64 bg-gray-900 border-l border-gray-800 flex items-center justify-center p-4">
        <p className="text-gray-500 text-sm text-center">Select a node or edge to edit its properties</p>
      </div>
    )
  }

  const handleNodeChange = (field, value) => {
    if (!selectedNode) return
    const step = { ...(selectedNode.data?.step || {}), [field]: value }
    onChange(selectedNode.id, { data: { ...selectedNode.data, step, label: field === 'name' ? value : selectedNode.data.label } })
  }

  const handleConfigChange = (field, value) => {
    if (!selectedNode) return
    const config = { ...(selectedNode.data?.step?.config || {}), [field]: value }
    handleNodeChange('config', config)
  }

  const stepType = selectedNode?.data?.step?.step_type || selectedNode?.type?.replace('Node', '')

  return (
    <div className="w-64 bg-gray-900 border-l border-gray-800 overflow-y-auto p-4 flex flex-col gap-4">
      <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
        {selectedEdge ? 'Edge Properties' : 'Node Properties'}
      </div>

      {selectedNode && (
        <>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-400">Label</label>
            <input
              className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white focus:border-indigo-500 focus:outline-none"
              value={selectedNode.data?.step?.name || selectedNode.data?.label || ''}
              onChange={e => handleNodeChange('name', e.target.value)}
            />
          </div>

          {(stepType === 'action' || stepType === 'actionNode') && (
            <>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Tool</label>
                <select
                  className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white focus:border-indigo-500 focus:outline-none"
                  value={selectedNode.data?.step?.tool_name || ''}
                  onChange={e => handleNodeChange('tool_name', e.target.value)}
                >
                  <option value="">Select tool...</option>
                  {tools.map(t => <option key={t.name || t} value={t.name || t}>{t.name || t}</option>)}
                </select>
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Persona</label>
                <select
                  className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white focus:border-indigo-500 focus:outline-none"
                  value={selectedNode.data?.step?.persona_name || ''}
                  onChange={e => handleNodeChange('persona_name', e.target.value)}
                >
                  <option value="">Select persona...</option>
                  {personas.map(p => <option key={p.name || p} value={p.name || p}>{p.name || p}</option>)}
                </select>
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Tool Params (JSON)</label>
                <textarea
                  className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white font-mono focus:border-indigo-500 focus:outline-none min-h-[80px]"
                  value={JSON.stringify(selectedNode.data?.step?.tool_params || {}, null, 2)}
                  onChange={e => { try { handleNodeChange('tool_params', JSON.parse(e.target.value)) } catch {} }}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Timeout (seconds)</label>
                <input
                  type="number"
                  className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white focus:border-indigo-500 focus:outline-none"
                  value={selectedNode.data?.step?.timeout_seconds || ''}
                  onChange={e => handleNodeChange('timeout_seconds', parseInt(e.target.value) || null)}
                />
              </div>
            </>
          )}

          {(stepType === 'trigger' || stepType === 'triggerNode') && (
            <>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Trigger Type</label>
                <div className="flex flex-col gap-1">
                  {['manual', 'webhook', 'cron'].map(t => (
                    <label key={t} className="flex items-center gap-2 text-sm text-gray-200 cursor-pointer">
                      <input
                        type="radio"
                        name="trigger_type"
                        value={t}
                        checked={(selectedNode.data?.step?.config?.trigger_type || 'manual') === t}
                        onChange={() => handleConfigChange('trigger_type', t)}
                        className="text-indigo-500"
                      />
                      {t}
                    </label>
                  ))}
                </div>
              </div>
              {selectedNode.data?.step?.config?.trigger_type === 'cron' && (
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-gray-400">Cron Expression</label>
                  <input
                    className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white font-mono focus:border-indigo-500 focus:outline-none"
                    value={selectedNode.data?.step?.config?.cron_expression || ''}
                    onChange={e => handleConfigChange('cron_expression', e.target.value)}
                    placeholder="0 9 * * *"
                  />
                </div>
              )}
            </>
          )}

          {(stepType === 'loop' || stepType === 'loopNode') && (
            <>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Iterator</label>
                <input
                  className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white font-mono focus:border-indigo-500 focus:outline-none"
                  value={selectedNode.data?.step?.config?.iterator || ''}
                  onChange={e => handleConfigChange('iterator', e.target.value)}
                  placeholder="${items}"
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Max Iterations</label>
                <input
                  type="number"
                  className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white focus:border-indigo-500 focus:outline-none"
                  value={selectedNode.data?.step?.config?.max_iterations || ''}
                  onChange={e => handleConfigChange('max_iterations', parseInt(e.target.value) || null)}
                />
              </div>
            </>
          )}

          {(stepType === 'wait' || stepType === 'waitNode') && (
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-400">Duration</label>
              <input
                className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white focus:border-indigo-500 focus:outline-none"
                value={selectedNode.data?.step?.config?.duration || ''}
                onChange={e => handleConfigChange('duration', e.target.value)}
                placeholder="30s, 5m, 1h"
              />
            </div>
          )}

          {(stepType === 'human_approval' || stepType === 'approvalNode') && (
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-400">Approvers</label>
              <textarea
                className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white focus:border-indigo-500 focus:outline-none min-h-[60px]"
                value={(selectedNode.data?.step?.config?.approvers || []).join('\n')}
                onChange={e => handleConfigChange('approvers', e.target.value.split('\n').filter(Boolean))}
                placeholder="One email per line"
              />
            </div>
          )}
        </>
      )}

      {selectedEdge && (
        <>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-400">Edge Type</label>
            <select
              className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white focus:border-indigo-500 focus:outline-none"
              value={selectedEdge.type || 'default'}
              onChange={e => onChange(selectedEdge.id, { type: e.target.value })}
            >
              <option value="default">Sequence</option>
              <option value="conditionalEdge">Conditional</option>
              <option value="errorEdge">Error</option>
              <option value="loopBackEdge">Loop Back</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-400">Condition</label>
            <input
              className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white font-mono focus:border-indigo-500 focus:outline-none"
              value={selectedEdge.data?.edge?.condition || selectedEdge.label || ''}
              onChange={e => onChange(selectedEdge.id, { label: e.target.value, data: { ...selectedEdge.data, edge: { ...(selectedEdge.data?.edge || {}), condition: e.target.value } } })}
              placeholder="${result} == true"
            />
          </div>
        </>
      )}
    </div>
  )
}
