import { useState, useEffect, useRef } from 'react'

export function SearchInput({ value, onChange, placeholder = 'Search...' }) {
  const [local, setLocal] = useState(value || '')
  const timer = useRef(null)

  useEffect(() => {
    setLocal(value || '')
  }, [value])

  function handleChange(e) {
    const v = e.target.value
    setLocal(v)
    clearTimeout(timer.current)
    timer.current = setTimeout(() => onChange(v), 300)
  }

  function handleClear() {
    setLocal('')
    clearTimeout(timer.current)
    onChange('')
  }

  return (
    <div className="relative">
      <input
        type="text"
        value={local}
        onChange={handleChange}
        placeholder={placeholder}
        className="w-full pl-3 pr-8 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
      />
      {local && (
        <button
          onClick={handleClear}
          className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 text-sm"
        >
          x
        </button>
      )}
    </div>
  )
}
