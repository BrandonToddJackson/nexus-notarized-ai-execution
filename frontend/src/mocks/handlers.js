import { http, HttpResponse } from 'msw'
import { fixtures } from './data.js'

export const handlers = [
  // Auth
  http.post('/v1/auth/token', () => {
    return HttpResponse.json({ token: fixtures.token, tenant_id: 'test-tenant' })
  }),

  // Health
  http.get('/v1/health', () => {
    return HttpResponse.json({ status: 'ok', services: { database: 'ok', redis: 'ok', vector_store: 'ok', llm: 'ok' } })
  }),

  // Workflows
  http.get('/v2/workflows/templates', () => {
    return HttpResponse.json({ templates: [] })
  }),
  http.get('/v2/workflows', () => {
    return HttpResponse.json({ workflows: [fixtures.workflow], total: 1 })
  }),
  http.post('/v2/workflows/generate', () => {
    return HttpResponse.json(fixtures.workflow, { status: 201 })
  }),
  http.post('/v2/workflows/import', () => {
    return HttpResponse.json(fixtures.workflow, { status: 201 })
  }),
  http.post('/v2/workflows', () => {
    return HttpResponse.json(fixtures.workflow, { status: 201 })
  }),
  http.get('/v2/workflows/:id/versions', () => {
    return HttpResponse.json([{ version: 1, created_at: '2026-01-01T00:00:00Z' }])
  }),
  http.post('/v2/workflows/:id/activate', () => {
    return HttpResponse.json({ ...fixtures.workflow, status: 'active' })
  }),
  http.post('/v2/workflows/:id/pause', () => {
    return HttpResponse.json({ ...fixtures.workflow, status: 'paused' })
  }),
  http.post('/v2/workflows/:id/rollback/:version', () => {
    return HttpResponse.json(fixtures.workflow)
  }),
  http.post('/v2/workflows/:id/duplicate', () => {
    return HttpResponse.json({ ...fixtures.workflow, id: 'wf-copy', name: 'Test Workflow (copy)' })
  }),
  http.post('/v2/workflows/:id/explain', () => {
    return HttpResponse.json({ explanation: 'This workflow does...', audience: 'developer' })
  }),
  http.post('/v2/workflows/:id/run', () => {
    return HttpResponse.json({ execution_id: 'ex-1', mode: 'inline', status: 'completed' })
  }),
  http.get('/v2/workflows/:id', ({ params }) => {
    if (params.id === 'wf-active') return HttpResponse.json(fixtures.activeWorkflow)
    return HttpResponse.json(fixtures.workflow)
  }),
  http.put('/v2/workflows/:id', async ({ request }) => {
    const body = await request.json()
    return HttpResponse.json({ ...fixtures.workflow, ...body })
  }),
  http.patch('/v2/workflows/:id/status', async ({ request }) => {
    const body = await request.json()
    return HttpResponse.json({ ...fixtures.workflow, ...body })
  }),
  http.patch('/v2/workflows/:id', async ({ request }) => {
    const body = await request.json()
    return HttpResponse.json({ ...fixtures.workflow, ...body })
  }),
  http.delete('/v2/workflows/:id', () => {
    return HttpResponse.json({}, { status: 204 })
  }),

  // Executions
  http.get('/v2/executions/:id/pins', () => {
    return HttpResponse.json([])
  }),
  http.post('/v2/executions/:id/steps/:stepId/pin', () => {
    return HttpResponse.json({ pinned: true })
  }),
  http.delete('/v2/executions/:id/steps/:stepId/pin', () => {
    return HttpResponse.json({ pinned: false })
  }),
  http.post('/v2/executions/:id/retry', () => {
    return HttpResponse.json({ execution_id: 'ex-new', mode: 'inline', status: 'running' })
  }),
  http.get('/v2/executions/:id', () => {
    return HttpResponse.json(fixtures.execution)
  }),
  http.delete('/v2/executions/:id', () => {
    return HttpResponse.json({})
  }),
  http.get('/v2/executions', () => {
    return HttpResponse.json({ executions: [fixtures.execution], total: 1 })
  }),

  // Credentials
  http.get('/v2/credentials/types', () => {
    return HttpResponse.json({ types: ['api_key', 'oauth', 'basic_auth', 'bearer_token'] })
  }),
  http.post('/v2/credentials/test', () => {
    return HttpResponse.json({ success: true, message: 'Connection successful' })
  }),
  http.post('/v2/credentials/:id/test', () => {
    return HttpResponse.json({ success: true, message: 'Connection successful' })
  }),
  http.get('/v2/credentials/:id/peek', () => {
    return HttpResponse.json({ value: '...key', masked: true })
  }),
  http.post('/v2/credentials/:id/peek', () => {
    return HttpResponse.json({ value: '...key', masked: true })
  }),
  http.post('/v2/credentials/:id/rotate', () => {
    return HttpResponse.json(fixtures.credential)
  }),
  http.delete('/v2/credentials/:id', () => {
    return HttpResponse.json({})
  }),
  http.get('/v2/credentials', () => {
    return HttpResponse.json({ credentials: [fixtures.credential] })
  }),
  http.post('/v2/credentials', () => {
    return HttpResponse.json(fixtures.credential, { status: 201 })
  }),

  // Skills
  http.get('/v2/skills/:id/diff', () => {
    return HttpResponse.json({ diff: '', from_version: 1, to_version: 1 })
  }),
  http.get('/v2/skills/:id/invocations', () => {
    return HttpResponse.json({ invocations: [], total: 0 })
  }),
  http.post('/v2/skills/:id/duplicate', () => {
    return HttpResponse.json({ ...fixtures.skill, id: 'sk-copy', name: 'Research Skill (copy)' })
  }),
  http.get('/v2/skills/:id', () => {
    return HttpResponse.json(fixtures.skill)
  }),
  http.patch('/v2/skills/:id', () => {
    return HttpResponse.json({ ...fixtures.skill, version: 2, updated_at: '2026-01-02T00:00:00Z' })
  }),
  http.delete('/v2/skills/:id', () => {
    return HttpResponse.json({})
  }),
  http.get('/v2/skills', () => {
    return HttpResponse.json({ skills: [fixtures.skill], total: 1 })
  }),
  http.post('/v2/skills', () => {
    return HttpResponse.json(fixtures.skill, { status: 201 })
  }),

  // MCP Servers
  http.get('/v2/mcp/servers/:id/tools', () => {
    return HttpResponse.json({ tools: [{ name: 'read_file', description: 'Read a file' }] })
  }),
  http.post('/v2/mcp/servers/:id/test', () => {
    return HttpResponse.json({ success: true, message: 'Connected' })
  }),
  http.post('/v2/mcp/servers/:id/reconnect', () => {
    return HttpResponse.json({ status: 'connected' })
  }),
  http.delete('/v2/mcp/servers/:id', () => {
    return HttpResponse.json({})
  }),
  http.get('/v2/mcp/servers', () => {
    return HttpResponse.json({ servers: [fixtures.mcpServer] })
  }),
  http.post('/v2/mcp/servers', () => {
    return HttpResponse.json(fixtures.mcpServer, { status: 201 })
  }),

  // Triggers
  http.post('/v2/triggers/:id/enable', () => {
    return HttpResponse.json({ ...fixtures.trigger, enabled: true })
  }),
  http.post('/v2/triggers/:id/disable', () => {
    return HttpResponse.json({ ...fixtures.trigger, enabled: false })
  }),
  http.get('/v2/triggers/:id', () => {
    return HttpResponse.json(fixtures.trigger)
  }),
  http.put('/v2/triggers/:id', () => {
    return HttpResponse.json(fixtures.trigger)
  }),
  http.delete('/v2/triggers/:id', () => {
    return HttpResponse.json({})
  }),
  http.get('/v2/triggers', () => {
    return HttpResponse.json({ triggers: [fixtures.trigger] })
  }),
  http.post('/v2/triggers', () => {
    return HttpResponse.json(fixtures.trigger, { status: 201 })
  }),

  // Ledger / Seals
  http.get('/v2/ledger/seals', () => {
    return HttpResponse.json({ seals: [fixtures.seal], total: 1 })
  }),

  // Marketplace
  http.get('/v2/marketplace/search', () => {
    return HttpResponse.json({ results: [] })
  }),
  http.get('/v2/marketplace/installed', () => {
    return HttpResponse.json({ plugins: [] })
  }),
]
