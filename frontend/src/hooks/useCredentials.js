import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { listCredentials, createCredential, deleteCredential, getCredentialTypes, testCredential } from '../api/credentials.js'

export const useCredentials = () => useQuery({
  queryKey: ['credentials'],
  queryFn: listCredentials,
  staleTime: 30_000,
})

export const useCredentialTypes = () => useQuery({
  queryKey: ['credential-types'],
  queryFn: getCredentialTypes,
  staleTime: 60_000,
})

export const useCreateCredential = () => {
  const qc = useQueryClient()
  return useMutation({ mutationFn: createCredential, onSuccess: () => qc.invalidateQueries({ queryKey: ['credentials'] }) })
}

export const useDeleteCredential = () => {
  const qc = useQueryClient()
  return useMutation({ mutationFn: deleteCredential, onSuccess: () => qc.invalidateQueries({ queryKey: ['credentials'] }) })
}

export const useTestCredential = () => {
  return useMutation({ mutationFn: testCredential })
}
