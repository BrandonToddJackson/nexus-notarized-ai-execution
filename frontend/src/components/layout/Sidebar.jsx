import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useAppStore } from '../../stores/appStore.js'
import { clearAuth } from '../../lib/auth.js'
import api from '../../lib/api.js'

const NAV_SECTIONS = [
  { items: [
    { label: 'Dashboard', path: '/', icon: '\uD83D\uDCCA' },
    { label: 'Execute', path: '/execute', icon: '\u26A1' },
  ]},
  { title: 'Automation', items: [
    { label: 'Workflows', path: '/workflows', icon: '\uD83D\uDD04' },
    { label: 'Executions', path: '/executions', icon: '\uD83D\uDCCB' },
  ]},
  { title: 'Connections', items: [
    { label: 'Credentials', path: '/credentials', icon: '\uD83D\uDD11' },
    { label: 'MCP Servers', path: '/mcp-servers', icon: '\uD83D\uDD0C' },
  ]},
  { title: 'Build', items: [
    { label: 'Skills', path: '/skills', icon: '\u26A1' },
    { label: 'Tools', path: '/tools', icon: '\uD83D\uDD27' },
    { label: 'Personas', path: '/personas', icon: '\uD83C\uDFAD' },
    { label: 'Knowledge', path: '/knowledge', icon: '\uD83D\uDCDA' },
  ]},
]

export function Sidebar() {
  const location = useLocation()
  const navigate = useNavigate()
  const { sidebarCollapsed, toggleSidebar } = useAppStore()

  function handleLogout() {
    clearAuth()
    api.clearToken()
    navigate('/login')
  }

  return (
    <aside
      className={`fixed left-0 top-0 bottom-0 bg-gray-900 text-white flex flex-col transition-all z-30 hidden md:flex ${
        sidebarCollapsed ? 'w-16' : 'w-60'
      }`}
    >
      <div className="p-4 border-b border-gray-800">
        <Link to="/" className="flex items-center gap-2">
          <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center font-bold text-sm shrink-0">N</div>
          {!sidebarCollapsed && <span className="text-lg font-bold tracking-wide">NEXUS</span>}
        </Link>
      </div>

      <nav className="flex-1 py-3 px-2 space-y-4 overflow-y-auto">
        {NAV_SECTIONS.map((section, si) => (
          <div key={si}>
            {!sidebarCollapsed && section.title && (
              <div className="px-3 mb-1 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                {section.title}
              </div>
            )}
            <div className="space-y-0.5">
              {section.items.map(({ path, label, icon }) => {
                const active = path === '/' ? location.pathname === '/' : location.pathname.startsWith(path)
                return (
                  <Link
                    key={path}
                    to={path}
                    title={sidebarCollapsed ? label : undefined}
                    className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                      active ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-800'
                    }`}
                  >
                    <span className="text-base shrink-0">{icon}</span>
                    {!sidebarCollapsed && label}
                  </Link>
                )
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="p-2 border-t border-gray-800 space-y-1">
        <button
          onClick={handleLogout}
          className="flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium text-gray-400 hover:text-red-400 hover:bg-gray-800 w-full"
        >
          <span className="text-base shrink-0">{'\uD83D\uDEAA'}</span>
          {!sidebarCollapsed && 'Logout'}
        </button>
        <button
          onClick={toggleSidebar}
          className="flex items-center justify-center w-full py-1 text-gray-500 hover:text-gray-300 text-xs"
        >
          {sidebarCollapsed ? '\u25B6' : '\u25C0'}
        </button>
      </div>
    </aside>
  )
}
