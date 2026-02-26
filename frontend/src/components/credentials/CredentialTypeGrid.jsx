import { useState } from 'react'

export function CredentialTypeGrid({ types = [], onSelect }) {
  const [search, setSearch] = useState('')
  const filtered = types.filter(t =>
    t.name?.toLowerCase().includes(search.toLowerCase()) ||
    t.category?.toLowerCase().includes(search.toLowerCase())
  )
  const grouped = {}
  filtered.forEach(t => {
    const cat = t.category || 'Other'
    if (!grouped[cat]) grouped[cat] = []
    grouped[cat].push(t)
  })

  return (
    <div>
      <input
        type="text"
        value={search}
        onChange={e => setSearch(e.target.value)}
        placeholder="Search credential types..."
        className="w-full border border-gray-300 rounded px-3 py-2 text-sm mb-4"
      />
      {Object.entries(grouped).map(([category, items]) => (
        <div key={category} className="mb-4">
          <h4 className="text-xs font-semibold text-gray-500 uppercase mb-2">{category}</h4>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {items.map(t => (
              <button
                key={t.type || t.name}
                onClick={() => onSelect(t)}
                className="text-left p-3 border rounded-lg hover:border-indigo-300 hover:bg-indigo-50 transition-colors"
              >
                <div className="text-sm font-medium text-gray-900">{t.name}</div>
                {t.description && <div className="text-xs text-gray-500 mt-1">{t.description}</div>}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
