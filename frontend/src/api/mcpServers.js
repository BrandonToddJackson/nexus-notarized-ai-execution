import api from '../lib/api.js'

export const listMCPServers = () => api.get('/v2/mcp/servers')
export const addMCPServer = (data) => api.post('/v2/mcp/servers', data)
export const removeMCPServer = (id) => api.delete(`/v2/mcp/servers/${id}`)
export const getMCPServerTools = (id) => api.get(`/v2/mcp/servers/${id}/tools`)
export const testMCPServer = (id) => api.post(`/v2/mcp/servers/${id}/test`)
export const reconnectMCPServer = (id) => api.post(`/v2/mcp/servers/${id}/reconnect`)
