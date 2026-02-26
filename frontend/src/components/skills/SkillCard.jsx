import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useUpdateSkill, useDeleteSkill, useDuplicateSkill } from '../../hooks/useSkills.js'
import { Toggle } from '../shared/Toggle.jsx'

export function SkillCard({ skill }) {
  const navigate = useNavigate()
  const [menuOpen, setMenuOpen] = useState(false)
  const updateSkill = useUpdateSkill()
  const deleteSkill = useDeleteSkill()
  const duplicateSkill = useDuplicateSkill()

  function handleToggle(active) {
    updateSkill.mutate({ id: skill.id, is_active: active })
  }

  return (
    <div className="bg-white border rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-2">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-gray-900 truncate">{skill.display_name || skill.name}</h3>
          <p className="text-xs text-gray-500 font-mono">{skill.name}</p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded">v{skill.version || 1}</span>
          <div className="relative">
            <button onClick={() => setMenuOpen(!menuOpen)} className="text-gray-400 hover:text-gray-600 px-1">
              ...
            </button>
            {menuOpen && (
              <div className="absolute right-0 top-6 bg-white border rounded-lg shadow-lg py-1 w-40 z-10">
                <button onClick={() => { navigate(`/skills/${skill.id}`); setMenuOpen(false) }} className="w-full text-left px-3 py-1.5 text-sm hover:bg-gray-50">Edit</button>
                <button onClick={() => { duplicateSkill.mutate(skill.id); setMenuOpen(false) }} className="w-full text-left px-3 py-1.5 text-sm hover:bg-gray-50">Duplicate</button>
                <button onClick={() => { navigate(`/skills/${skill.id}?tab=invocations`); setMenuOpen(false) }} className="w-full text-left px-3 py-1.5 text-sm hover:bg-gray-50">View Invocations</button>
                <button onClick={() => { setMenuOpen(false) }} className="w-full text-left px-3 py-1.5 text-sm hover:bg-gray-50">Export</button>
                <hr className="my-1" />
                <button onClick={() => { deleteSkill.mutate(skill.id); setMenuOpen(false) }} className="w-full text-left px-3 py-1.5 text-sm text-red-600 hover:bg-red-50">Delete</button>
              </div>
            )}
          </div>
        </div>
      </div>
      {skill.description && (
        <p className="text-xs text-gray-600 mb-3 line-clamp-2">
          {skill.description.length > 120 ? skill.description.slice(0, 120) + '...' : skill.description}
        </p>
      )}
      <div className="flex items-center justify-between">
        <div className="flex gap-1 flex-wrap">
          {skill.tags?.slice(0, 3).map(tag => (
            <span key={tag} className="text-xs bg-indigo-50 text-indigo-600 px-1.5 py-0.5 rounded">{tag}</span>
          ))}
        </div>
        <div className="flex items-center gap-3">
          {skill.invocation_count != null && (
            <span className="text-xs text-gray-400">{skill.invocation_count} invocations</span>
          )}
          <Toggle checked={skill.is_active !== false} onChange={handleToggle} />
        </div>
      </div>
    </div>
  )
}
