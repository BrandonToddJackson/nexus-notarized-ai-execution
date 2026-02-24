import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { listExecutions, getExecution, retryExecution } from '../api/executions.js'

export const useExecutions = (filters) => useQuery({
  queryKey: ['executions', filters],
  queryFn: () => listExecutions(filters),
  staleTime: 15_000,
})

export const useExecution = (id) => useQuery({
  queryKey: ['executions', id],
  queryFn: () => getExecution(id),
  enabled: !!id,
})

export const useRetryExecution = () => {
  const qc = useQueryClient()
  return useMutation({ mutationFn: retryExecution, onSuccess: () => qc.invalidateQueries({ queryKey: ['executions'] }) })
}
