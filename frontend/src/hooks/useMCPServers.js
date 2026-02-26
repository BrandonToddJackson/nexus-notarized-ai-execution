import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { listMCPServers, addMCPServer, removeMCPServer, getMCPServerTools, reconnectMCPServer, testMCPServer } from '../api/mcpServers.js'

export const useMCPServers = () => useQuery({
  queryKey: ['mcp-servers'],
  queryFn: listMCPServers,
  staleTime: 30_000,
  refetchInterval: 30_000,
})

export const useMCPServerTools = (id) => useQuery({
  queryKey: ['mcp-servers', id, 'tools'],
  queryFn: () => getMCPServerTools(id),
  enabled: !!id,
})

export const useAddMCPServer = () => {
  const qc = useQueryClient()
  return useMutation({ mutationFn: addMCPServer, onSuccess: () => qc.invalidateQueries({ queryKey: ['mcp-servers'] }) })
}

export const useRemoveMCPServer = () => {
  const qc = useQueryClient()
  return useMutation({ mutationFn: removeMCPServer, onSuccess: () => qc.invalidateQueries({ queryKey: ['mcp-servers'] }) })
}

export const useReconnectMCPServer = () => {
  const qc = useQueryClient()
  return useMutation({ mutationFn: reconnectMCPServer, onSuccess: () => qc.invalidateQueries({ queryKey: ['mcp-servers'] }) })
}

export const useTestMCPServer = () => {
  return useMutation({ mutationFn: testMCPServer })
}
