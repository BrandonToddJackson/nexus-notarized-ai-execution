/**
 * Auth utilities: JWT storage, refresh, tenant context.
 */

export function getToken() {
  return localStorage.getItem('nexus_token')
}

export function setToken(token) {
  localStorage.setItem('nexus_token', token)
}

export function clearAuth() {
  localStorage.removeItem('nexus_token')
  localStorage.removeItem('nexus_tenant')
}

export function isAuthenticated() {
  return !!getToken()
}
