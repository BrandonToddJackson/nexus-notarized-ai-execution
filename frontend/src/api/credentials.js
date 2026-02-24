import api from '../lib/api.js'

export const listCredentials = () => api.get('/v2/credentials')
export const createCredential = (data) => api.post('/v2/credentials', data)
export const deleteCredential = (id) => api.delete(`/v2/credentials/${id}`)
export const getCredentialTypes = () => api.get('/v2/credentials/types')
export const testCredential = (id) => api.post(`/v2/credentials/${id}/test`)
export const getCredentialUsage = (id) => api.get(`/v2/credentials/${id}/usage`)
export const peekCredential = (id) => api.get(`/v2/credentials/${id}/peek`)
export const rotateCredential = (id, data) => api.post(`/v2/credentials/${id}/rotate`, data)
