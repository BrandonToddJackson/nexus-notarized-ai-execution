/**
 * Auth utilities: JWT storage, refresh, tenant context.
 *
 * Security model:
 * - JWT stored in sessionStorage (not localStorage) — cleared on tab/browser close,
 *   reducing the XSS persistence window. The token is never written to disk.
 * - The raw API key is never stored — only the short-lived JWT returned after exchange.
 * - Production deployments should upgrade to httpOnly cookies (requires backend support)
 *   so the token is fully inaccessible to JavaScript.
 */

const TOKEN_KEY = 'nexus_token'
const TENANT_KEY = 'nexus_tenant'

export function getToken() {
  return sessionStorage.getItem(TOKEN_KEY)
}

export function setToken(token) {
  sessionStorage.setItem(TOKEN_KEY, token)
}

export function getTenantId() {
  return sessionStorage.getItem(TENANT_KEY)
}

export function setTenantId(tenantId) {
  sessionStorage.setItem(TENANT_KEY, tenantId)
}

export function clearAuth() {
  sessionStorage.removeItem(TOKEN_KEY)
  sessionStorage.removeItem(TENANT_KEY)
}

export function isAuthenticated() {
  return !!getToken()
}
