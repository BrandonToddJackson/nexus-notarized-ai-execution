import api from '../lib/api.js'

export const listSkills = (params) => {
  const qs = params ? '?' + new URLSearchParams(
    Object.fromEntries(Object.entries(params).filter(([, v]) => v != null))
  ).toString() : ''
  return api.get(`/v2/skills${qs}`)
}
export const createSkill = (data) => api.post('/v2/skills', data)
export const getSkill = (id) => api.get(`/v2/skills/${id}`)
export const updateSkill = (id, data) => api.patch(`/v2/skills/${id}`, data)
export const deleteSkill = (id) => api.delete(`/v2/skills/${id}`)
export const exportSkill = (id) => api.get(`/v2/skills/${id}/export`)
export const importSkill = (data) => api.post('/v2/skills/import', data)
export const getSkillInvocations = (id, params) => {
  const qs = params ? '?' + new URLSearchParams(params).toString() : ''
  return api.get(`/v2/skills/${id}/invocations${qs}`)
}
export const duplicateSkill = (id) => api.post(`/v2/skills/${id}/duplicate`)
export const diffSkillVersions = (id, from, to) =>
  api.get(`/v2/skills/${id}/diff?from=${from}&to=${to}`)
