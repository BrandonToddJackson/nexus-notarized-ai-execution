import 'reactflow/dist/style.css'
import '../styles/workflow-editor.css'

import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  MarkerType,
} from 'reactflow'
import { useRef, useState, useCallback, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

import TriggerNode from '../components/workflow/nodes/TriggerNode'
import ActionNode from '../components/workflow/nodes/ActionNode'
import BranchNode from '../components/workflow/nodes/BranchNode'
import LoopNode from '../components/workflow/nodes/LoopNode'
import ParallelNode from '../components/workflow/nodes/ParallelNode'
import SubWorkflowNode from '../components/workflow/nodes/SubWorkflowNode'
import WaitNode from '../components/workflow/nodes/WaitNode'
import ApprovalNode from '../components/workflow/nodes/ApprovalNode'

import ConditionalEdge from '../components/workflow/edges/ConditionalEdge'
import ErrorEdge from '../components/workflow/edges/ErrorEdge'
import LoopBackEdge from '../components/workflow/edges/LoopBackEdge'

import NodeToolbox from '../components/workflow/panels/NodeToolbox'
import PropertiesPanel from '../components/workflow/panels/PropertiesPanel'
import VersionHistoryPanel from '../components/workflow/panels/VersionHistoryPanel'

import NLGenerateModal from '../components/workflow/modals/NLGenerateModal'
import ValidationModal from '../components/workflow/modals/ValidationModal'

import GateStatusOverlay from '../components/workflow/overlays/GateStatusOverlay'

import { workflowToFlow } from '../components/workflow/utils/workflow-to-flow'
import { flowToWorkflow } from '../components/workflow/utils/flow-to-workflow'
import { applyDagreLayout } from '../components/workflow/utils/dagre-layout'

import api from '../lib/api'

const nodeTypes = {
  triggerNode: TriggerNode,
  actionNode: ActionNode,
  branchNode: BranchNode,
  loopNode: LoopNode,
  parallelNode: ParallelNode,
  subWorkflowNode: SubWorkflowNode,
  waitNode: WaitNode,
  approvalNode: ApprovalNode,
}

const edgeTypes = {
  conditionalEdge: ConditionalEdge,
  errorEdge: ErrorEdge,
  loopBackEdge: LoopBackEdge,
}

const STEP_TYPE_TO_NODE_TYPE = {
  action: 'actionNode',
  branch: 'branchNode',
  loop: 'loopNode',
  parallel: 'parallelNode',
  sub_workflow: 'subWorkflowNode',
  wait: 'waitNode',
  human_approval: 'approvalNode',
  trigger: 'triggerNode',
}

export default function WorkflowEditor() {
  const { workflowId } = useParams()
  const navigate = useNavigate()
  const reactFlowWrapper = useRef(null)
  const [reactFlowInstance, setReactFlowInstance] = useState(null)

  const [workflow, setWorkflow] = useState(null)
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [selectedNode, setSelectedNode] = useState(null)
  const [selectedEdge, setSelectedEdge] = useState(null)
  const [isDirty, setIsDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [loading, setLoading] = useState(true)
  const [showNLModal, setShowNLModal] = useState(false)
  const [showValidateModal, setShowValidateModal] = useState(false)
  const [showVersionHistory, setShowVersionHistory] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (workflowId === 'new') {
      setNodes([
        {
          id: 'trigger_1',
          type: 'triggerNode',
          position: { x: 250, y: 50 },
          data: {
            label: 'Trigger',
            step: { id: 'trigger_1', name: 'Trigger', step_type: 'trigger', config: { trigger_type: 'manual' } },
          },
        },
      ])
      setLoading(false)
      return
    }
    api.get(`/v2/workflows/${workflowId}`)
      .then((data) => {
        setWorkflow(data)
        const { nodes: n, edges: e } = workflowToFlow(data)
        const laid = applyDagreLayout(n, e)
        setNodes(laid)
        setEdges(e)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [workflowId])

  const onConnect = useCallback(
    (params) => {
      setEdges((eds) => addEdge({ ...params, type: 'default', markerEnd: { type: MarkerType.ArrowClosed } }, eds))
      setIsDirty(true)
    },
    [setEdges],
  )

  const onDragOver = useCallback((event) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback(
    (event) => {
      event.preventDefault()
      const type = event.dataTransfer.getData('application/reactflow')
      if (!type || !reactFlowInstance || !reactFlowWrapper.current) return

      const rect = reactFlowWrapper.current.getBoundingClientRect()
      const position = reactFlowInstance.project({
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      })

      const newNode = {
        id: `node_${Date.now()}`,
        type: STEP_TYPE_TO_NODE_TYPE[type] || type + 'Node',
        position,
        data: {
          label: type,
          step: { id: `step_${Date.now()}`, name: type, step_type: type, tool_name: '', config: {} },
        },
      }
      setNodes((ns) => [...ns, newNode])
      setIsDirty(true)
    },
    [reactFlowInstance, setNodes],
  )

  const onNodeClick = useCallback((_, node) => {
    setSelectedNode(node)
    setSelectedEdge(null)
  }, [])

  const onEdgeClick = useCallback((_, edge) => {
    setSelectedEdge(edge)
    setSelectedNode(null)
  }, [])

  const onPaneClick = useCallback(() => {
    setSelectedNode(null)
    setSelectedEdge(null)
  }, [])

  const handlePropertiesChange = useCallback(
    (nodeId, updatedData) => {
      setNodes((ns) => ns.map((n) => (n.id === nodeId ? { ...n, ...updatedData } : n)))
      setIsDirty(true)
    },
    [setNodes],
  )

  const handleSave = useCallback(async () => {
    setSaving(true)
    setError(null)
    try {
      const dto = flowToWorkflow(nodes, edges, workflow)
      if (workflowId === 'new') {
        const result = await api.post('/v2/workflows', dto)
        navigate(`/workflows/${result.id}`)
      } else {
        await api.put(`/v2/workflows/${workflowId}`, dto)
      }
      setIsDirty(false)
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }, [nodes, edges, workflow, workflowId, navigate])

  const handleStatusToggle = useCallback(async () => {
    const current = workflow?.status
    const target = current === 'active' ? 'paused' : 'active'
    try {
      await api.patch(`/v2/workflows/${workflowId}/status`, { status: target })
      setWorkflow((w) => ({ ...w, status: target }))
    } catch (e) {
      setError(e.message)
    }
  }, [workflow, workflowId])

  const handleAutoLayout = useCallback(() => {
    setNodes((ns) => applyDagreLayout(ns, edges))
  }, [edges, setNodes])

  const handleNLGenerated = useCallback(
    (dto) => {
      const { nodes: n, edges: e } = workflowToFlow(dto)
      setNodes(applyDagreLayout(n, e))
      setEdges(e)
      setWorkflow(dto)
      setShowNLModal(false)
      setIsDirty(true)
    },
    [setNodes, setEdges],
  )

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-950">
        <div className="text-gray-400 text-sm">Loading workflow...</div>
      </div>
    )
  }

  if (error && !workflow && workflowId !== 'new') {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-950">
        <div className="text-red-400 text-sm">{error}</div>
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col bg-gray-950">
      {/* Toolbar */}
      <div className="h-14 bg-gray-900 border-b border-gray-800 flex items-center px-4 gap-3 shrink-0">
        <button
          onClick={() => navigate('/workflows')}
          className="text-gray-400 hover:text-white transition-colors"
          title="Back to workflows"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>

        <span className="text-white font-medium text-sm truncate max-w-[200px]">
          {workflow?.name || 'Untitled Workflow'}
        </span>

        <span className="text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded">
          v{workflow?.version || 1}
        </span>

        {isDirty && <div className="w-2 h-2 rounded-full bg-yellow-400" title="Unsaved changes" />}

        <div className="flex-1" />

        <button
          onClick={handleAutoLayout}
          className="px-3 py-1.5 text-xs text-gray-300 bg-gray-800 hover:bg-gray-700 rounded transition-colors"
        >
          Auto Layout
        </button>

        <button
          onClick={() => setShowValidateModal(true)}
          className="px-3 py-1.5 text-xs text-gray-300 bg-gray-800 hover:bg-gray-700 rounded transition-colors"
        >
          Validate
        </button>

        <button
          onClick={() => setShowNLModal(true)}
          className="px-3 py-1.5 text-xs text-indigo-300 bg-indigo-900/50 hover:bg-indigo-900 rounded transition-colors"
        >
          Generate
        </button>

        <button
          onClick={handleSave}
          disabled={saving}
          className="px-3 py-1.5 text-xs text-white bg-indigo-600 hover:bg-indigo-500 rounded disabled:opacity-50 transition-colors"
        >
          {saving ? 'Saving...' : 'Save'}
        </button>

        {workflowId !== 'new' && (
          <button
            onClick={handleStatusToggle}
            className={`px-3 py-1.5 text-xs rounded transition-colors ${
              workflow?.status === 'active'
                ? 'text-yellow-300 bg-yellow-900/50 hover:bg-yellow-900'
                : 'text-green-300 bg-green-900/50 hover:bg-green-900'
            }`}
          >
            {workflow?.status === 'active' ? 'Pause' : 'Activate'}
          </button>
        )}

        <button
          onClick={() => setShowVersionHistory(true)}
          className="text-gray-400 hover:text-white transition-colors"
          title="Version history"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </button>
      </div>

      {/* Main area */}
      <div className="flex-1 flex overflow-hidden">
        <NodeToolbox />

        <div className="flex-1 relative" ref={reactFlowWrapper}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={setReactFlowInstance}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onNodeClick={onNodeClick}
            onEdgeClick={onEdgeClick}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
          >
            <Background color="#374151" gap={20} />
            <Controls className="workflow-controls" />
            <MiniMap
              nodeColor={(n) => {
                if (n.type === 'triggerNode') return '#4f46e5'
                if (n.type === 'branchNode') return '#eab308'
                if (n.type === 'loopNode') return '#06b6d4'
                if (n.type === 'approvalNode') return '#f97316'
                return '#6b7280'
              }}
              style={{ background: '#111827' }}
            />
          </ReactFlow>

          <GateStatusOverlay />
        </div>

        <PropertiesPanel
          selectedNode={selectedNode}
          selectedEdge={selectedEdge}
          onChange={handlePropertiesChange}
        />
      </div>

      {/* Modals */}
      {showNLModal && (
        <NLGenerateModal onGenerated={handleNLGenerated} onClose={() => setShowNLModal(false)} />
      )}

      {showValidateModal && (
        <ValidationModal
          nodes={nodes}
          edges={edges}
          onSaveAnyway={() => {
            setShowValidateModal(false)
            handleSave()
          }}
          onClose={() => setShowValidateModal(false)}
        />
      )}

      {showVersionHistory && (
        <VersionHistoryPanel workflow={workflow} onClose={() => setShowVersionHistory(false)} />
      )}

      {error && (
        <div className="absolute bottom-4 right-4 bg-red-900/90 border border-red-700 rounded-lg px-4 py-2 text-sm text-red-200 z-50">
          {error}
          <button onClick={() => setError(null)} className="ml-3 text-red-400 hover:text-white">
            Dismiss
          </button>
        </div>
      )}
    </div>
  )
}
