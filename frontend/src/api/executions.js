import api from '../lib/api.js'

export const listExecutions = (params) => {
  const qs = params ? '?' + new URLSearchParams(
    Object.fromEntries(Object.entries(params).filter(([, v]) => v != null))
  ).toString() : ''
  return api.get(`/v2/executions${qs}`)
}
export const getExecution = (id) => api.get(`/v2/executions/${id}`)
export const retryExecution = (id) => api.post(`/v2/executions/${id}/retry`)
export const getExecutionPins = (id) => api.get(`/v2/executions/${id}/pins`)
export const pinStepOutput = (id, stepId) => api.post(`/v2/executions/${id}/steps/${stepId}/pin`)
export const unpinStepOutput = (id, stepId) => api.delete(`/v2/executions/${id}/steps/${stepId}/pin`)
