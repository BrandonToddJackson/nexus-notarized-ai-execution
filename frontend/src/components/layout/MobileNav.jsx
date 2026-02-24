import { Link, useLocation } from 'react-router-dom'

const TABS = [
  { label: 'Dashboard', path: '/', icon: '\uD83D\uDCCA' },
  { label: 'Execute', path: '/execute', icon: '\u26A1' },
  { label: 'Workflows', path: '/workflows', icon: '\uD83D\uDD04' },
  { label: 'Skills', path: '/skills', icon: '\u26A1' },
]

export function MobileNav() {
  const location = useLocation()

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-white border-t flex md:hidden z-30">
      {TABS.map(({ label, path, icon }) => {
        const active = path === '/' ? location.pathname === '/' : location.pathname.startsWith(path)
        return (
          <Link
            key={path}
            to={path}
            className={`flex-1 flex flex-col items-center py-2 text-xs ${
              active ? 'text-indigo-600' : 'text-gray-500'
            }`}
          >
            <span className="text-lg">{icon}</span>
            {label}
          </Link>
        )
      })}
    </nav>
  )
}
