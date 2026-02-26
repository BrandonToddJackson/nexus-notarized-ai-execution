import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { listWorkflows, createWorkflow, getWorkflow, updateWorkflow, deleteWorkflow, getTemplates, createFromTemplate } from '../api/workflows.js'

export const useWorkflows = () => useQuery({
  queryKey: ['workflows'],
  queryFn: listWorkflows,
  staleTime: 30_000,
})

export const useWorkflow = (id) => useQuery({
  queryKey: ['workflows', id],
  queryFn: () => getWorkflow(id),
  enabled: !!id,
})

export const useCreateWorkflow = () => {
  const qc = useQueryClient()
  return useMutation({ mutationFn: createWorkflow, onSuccess: () => qc.invalidateQueries({ queryKey: ['workflows'] }) })
}

export const useUpdateWorkflow = () => {
  const qc = useQueryClient()
  return useMutation({ mutationFn: ({ id, ...data }) => updateWorkflow(id, data), onSuccess: () => qc.invalidateQueries({ queryKey: ['workflows'] }) })
}

export const useDeleteWorkflow = () => {
  const qc = useQueryClient()
  return useMutation({ mutationFn: deleteWorkflow, onSuccess: () => qc.invalidateQueries({ queryKey: ['workflows'] }) })
}

export const useWorkflowTemplates = () => useQuery({
  queryKey: ['workflow-templates'],
  queryFn: getTemplates,
  staleTime: 60_000,
})

export const useCreateFromTemplate = () => {
  const qc = useQueryClient()
  return useMutation({ mutationFn: createFromTemplate, onSuccess: () => qc.invalidateQueries({ queryKey: ['workflows'] }) })
}
