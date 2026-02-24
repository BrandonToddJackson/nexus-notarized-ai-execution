import { useWorkflowTemplates, useCreateFromTemplate } from '../../hooks/useWorkflows.js'
import { useNavigate } from 'react-router-dom'

export function TemplateGallery({ onClose }) {
  const navigate = useNavigate()
  const { data: templates, isLoading } = useWorkflowTemplates()
  const createFromTemplate = useCreateFromTemplate()

  function handleUse(templateId) {
    createFromTemplate.mutate(templateId, {
      onSuccess: (data) => {
        onClose()
        if (data?.id) navigate(`/workflows/${data.id}/edit`)
      }
    })
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] flex flex-col" onClick={e => e.stopPropagation()}>
        <div className="px-6 py-4 border-b flex items-center justify-between">
          <h3 className="text-lg font-semibold">Workflow Templates</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">x</button>
        </div>
        <div className="p-6 overflow-y-auto">
          {isLoading ? (
            <div className="text-center text-gray-500 py-8">Loading templates...</div>
          ) : !templates?.length ? (
            <div className="text-center text-gray-500 py-8">No templates available</div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {templates.map(t => (
                <div key={t.id} className="border rounded-lg p-4">
                  <h4 className="text-sm font-medium text-gray-900 mb-1">{t.name}</h4>
                  {t.description && <p className="text-xs text-gray-500 mb-3">{t.description}</p>}
                  <button
                    onClick={() => handleUse(t.id)}
                    disabled={createFromTemplate.isPending}
                    className="text-xs px-3 py-1.5 bg-indigo-600 text-white rounded hover:bg-indigo-500 disabled:opacity-50"
                  >
                    Use template
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
