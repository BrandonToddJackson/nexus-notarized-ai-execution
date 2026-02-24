import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { listSkills, createSkill, getSkill, updateSkill, deleteSkill, getSkillInvocations, duplicateSkill } from '../api/skills.js'

export const useSkills = (filters) => useQuery({
  queryKey: ['skills', filters],
  queryFn: () => listSkills(filters),
  staleTime: 30_000,
})

export const useSkill = (id) => useQuery({
  queryKey: ['skills', id],
  queryFn: () => getSkill(id),
  enabled: !!id,
})

export const useSkillInvocations = (id) => useQuery({
  queryKey: ['skills', id, 'invocations'],
  queryFn: () => getSkillInvocations(id),
  enabled: !!id,
})

export const useCreateSkill = () => {
  const qc = useQueryClient()
  return useMutation({ mutationFn: createSkill, onSuccess: () => qc.invalidateQueries({ queryKey: ['skills'] }) })
}

export const useUpdateSkill = () => {
  const qc = useQueryClient()
  return useMutation({ mutationFn: ({ id, ...data }) => updateSkill(id, data), onSuccess: () => qc.invalidateQueries({ queryKey: ['skills'] }) })
}

export const useDeleteSkill = () => {
  const qc = useQueryClient()
  return useMutation({ mutationFn: deleteSkill, onSuccess: () => qc.invalidateQueries({ queryKey: ['skills'] }) })
}

export const useDuplicateSkill = () => {
  const qc = useQueryClient()
  return useMutation({ mutationFn: duplicateSkill, onSuccess: () => qc.invalidateQueries({ queryKey: ['skills'] }) })
}
