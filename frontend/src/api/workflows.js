import api from '../lib/api.js'

export const listWorkflows = () => api.get('/v2/workflows')
export const createWorkflow = (data) => api.post('/v2/workflows', data)
export const getWorkflow = (id) => api.get(`/v2/workflows/${id}`)
export const updateWorkflow = (id, data) => api.put(`/v2/workflows/${id}`, data)
export const patchWorkflow = (id, data) => api.patch(`/v2/workflows/${id}`, data)
export const deleteWorkflow = (id) => api.delete(`/v2/workflows/${id}`)
export const exportWorkflow = (id) => api.get(`/v2/workflows/${id}/export`)
export const importWorkflow = (data) => api.post('/v2/workflows/import', data)
export const duplicateWorkflow = (id) => api.post(`/v2/workflows/${id}/duplicate`)
export const getTemplates = () => api.get('/v2/workflows/templates')
export const createFromTemplate = (templateId) => api.post(`/v2/workflows/templates/${templateId}`)
