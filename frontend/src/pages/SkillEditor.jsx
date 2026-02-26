import { useParams, useNavigate } from 'react-router-dom'
import { useSkill, useUpdateSkill, useCreateSkill } from '../hooks/useSkills.js'
import { SkillEditorPanel } from '../components/skills/SkillEditorPanel.jsx'

export default function SkillEditor() {
  const { skillId } = useParams()
  const navigate = useNavigate()
  const isNew = !skillId || skillId === 'new'
  const { data: skill, isLoading } = useSkill(isNew ? null : skillId)
  const updateSkill = useUpdateSkill()
  const createSkill = useCreateSkill()

  function handleSave(form) {
    if (isNew) {
      createSkill.mutate(form, {
        onSuccess: (data) => {
          if (data?.id) navigate(`/skills/${data.id}`)
        }
      })
    } else {
      updateSkill.mutate({ id: skillId, ...form })
    }
  }

  if (!isNew && isLoading) {
    return <div className="p-6 text-gray-500">Loading...</div>
  }

  return (
    <div className="h-[calc(100vh-theme(spacing.14)-theme(spacing.12))]">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/skills')} className="text-gray-500 hover:text-gray-700 text-sm">
            &larr; Back
          </button>
          <h1 className="text-xl font-bold text-gray-900">{isNew ? 'New Skill' : (skill?.display_name || skill?.name || 'Edit Skill')}</h1>
        </div>
      </div>
      <SkillEditorPanel
        skill={isNew ? null : skill}
        onSave={handleSave}
        saving={updateSkill.isPending || createSkill.isPending}
      />
    </div>
  )
}
