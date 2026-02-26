import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSkills } from '../hooks/useSkills.js'
import { SkillCard } from '../components/skills/SkillCard.jsx'
import { SearchInput } from '../components/shared/SearchInput.jsx'
import { Toggle } from '../components/shared/Toggle.jsx'
import { EmptyState } from '../components/shared/EmptyState.jsx'

export default function Skills() {
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [activeOnly, setActiveOnly] = useState(false)
  const { data: skills, isLoading } = useSkills({ search: search || undefined, active: activeOnly || undefined })
  const skillList = Array.isArray(skills) ? skills : skills?.skills || []

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Skills</h1>
        <div className="flex gap-2">
          <button className="px-4 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50">Import</button>
          <button
            onClick={() => navigate('/skills/new')}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg"
          >
            New Skill
          </button>
        </div>
      </div>
      <div className="flex gap-4 mb-4 items-center">
        <div className="w-64">
          <SearchInput value={search} onChange={setSearch} placeholder="Search skills..." />
        </div>
        <Toggle checked={activeOnly} onChange={setActiveOnly} label="Active only" />
      </div>
      {isLoading ? (
        <div className="text-gray-500">Loading...</div>
      ) : skillList.length === 0 ? (
        <EmptyState
          icon="\u26A1"
          title="No skills yet"
          description="Create your first skill to get started"
          actionLabel="New Skill"
          onAction={() => navigate('/skills/new')}
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {skillList.map(s => <SkillCard key={s.id} skill={s} />)}
        </div>
      )}
    </div>
  )
}
