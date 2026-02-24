import { useNavigate } from 'react-router-dom'
import { useAppStore } from '../../stores/appStore.js'

export function TopBar() {
  const navigate = useNavigate()
  const { toggleSidebar, gateFailureCount, clearGateFailures } = useAppStore()

  function handleBellClick() {
    clearGateFailures()
    navigate('/executions')
  }

  return (
    <header className="h-14 bg-white border-b flex items-center justify-between px-4 shrink-0">
      <button onClick={toggleSidebar} className="text-gray-500 hover:text-gray-700 md:hidden">
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
      <div className="flex-1" />
      <div className="flex items-center gap-3">
        <button onClick={handleBellClick} className="relative text-gray-500 hover:text-gray-700">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
          </svg>
          {gateFailureCount > 0 && (
            <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
              {gateFailureCount > 9 ? '9+' : gateFailureCount}
            </span>
          )}
        </button>
      </div>
    </header>
  )
}
