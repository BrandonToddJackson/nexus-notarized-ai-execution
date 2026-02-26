import { useState, useEffect, useRef, useCallback } from 'react'
import { EditorView } from '@codemirror/view'
import { EditorState } from '@codemirror/state'
import { markdown } from '@codemirror/lang-markdown'
import { basicSetup } from '@codemirror/basic-setup'

export function SkillEditorPanel({ skill, onSave, saving }) {
  const [form, setForm] = useState({
    name: '',
    display_name: '',
    description: '',
    tags: [],
    template: '',
    change_note: '',
  })
  const [tagInput, setTagInput] = useState('')
  const [nameError, setNameError] = useState('')
  const editorRef = useRef(null)
  const viewRef = useRef(null)

  useEffect(() => {
    if (skill) {
      setForm({
        name: skill.name || '',
        display_name: skill.display_name || '',
        description: skill.description || '',
        tags: skill.tags || [],
        template: skill.template || '',
        change_note: '',
      })
    }
  }, [skill])

  const updateTemplate = useCallback((val) => {
    setForm(f => ({ ...f, template: val }))
  }, [])

  useEffect(() => {
    if (!editorRef.current) return
    const state = EditorState.create({
      doc: form.template,
      extensions: [
        basicSetup,
        markdown(),
        EditorView.updateListener.of(update => {
          if (update.docChanged) {
            updateTemplate(update.state.doc.toString())
          }
        }),
      ],
    })
    const view = new EditorView({ state, parent: editorRef.current })
    viewRef.current = view
    return () => view.destroy()
  }, [skill?.id])

  function handleNameChange(e) {
    const val = e.target.value
    setForm(f => ({ ...f, name: val }))
    if (val && !/^[a-z0-9-]{1,64}$/.test(val)) {
      setNameError('Must be lowercase alphanumeric with hyphens, max 64 chars')
    } else {
      setNameError('')
    }
  }

  function handleDisplayNameChange(e) {
    const val = e.target.value
    setForm(f => ({
      ...f,
      display_name: val,
      name: f.name || val.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 64),
    }))
  }

  function addTag() {
    if (tagInput.trim() && !form.tags.includes(tagInput.trim())) {
      setForm(f => ({ ...f, tags: [...f.tags, tagInput.trim()] }))
      setTagInput('')
    }
  }

  function removeTag(tag) {
    setForm(f => ({ ...f, tags: f.tags.filter(t => t !== tag) }))
  }

  function handleSave() {
    if (nameError) return
    onSave(form)
  }

  return (
    <div className="flex h-full gap-6">
      <div className="w-80 shrink-0 space-y-4 overflow-y-auto">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Display Name</label>
          <input
            value={form.display_name}
            onChange={handleDisplayNameChange}
            className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Slug</label>
          <input
            value={form.name}
            onChange={handleNameChange}
            className={`w-full border rounded px-3 py-2 text-sm font-mono ${nameError ? 'border-red-300' : 'border-gray-300'}`}
          />
          {nameError && <p className="text-xs text-red-500 mt-1">{nameError}</p>}
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Description <span className="text-gray-400">({form.description.length}/500)</span>
          </label>
          <textarea
            value={form.description}
            onChange={e => setForm(f => ({ ...f, description: e.target.value.slice(0, 500) }))}
            rows={3}
            className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Tags</label>
          <div className="flex gap-1 flex-wrap mb-2">
            {form.tags.map(tag => (
              <span key={tag} className="text-xs bg-indigo-50 text-indigo-600 px-2 py-0.5 rounded flex items-center gap-1">
                {tag}
                <button onClick={() => removeTag(tag)} className="text-indigo-400 hover:text-indigo-700">x</button>
              </span>
            ))}
          </div>
          <div className="flex gap-1">
            <input
              value={tagInput}
              onChange={e => setTagInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && (e.preventDefault(), addTag())}
              placeholder="Add tag..."
              className="flex-1 border border-gray-300 rounded px-2 py-1 text-sm"
            />
            <button onClick={addTag} className="px-2 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50">+</button>
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Change Note (required)</label>
          <input
            value={form.change_note}
            onChange={e => setForm(f => ({ ...f, change_note: e.target.value }))}
            placeholder="What changed..."
            className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
          />
        </div>
        <button
          onClick={handleSave}
          disabled={saving || !!nameError || !form.change_note}
          className="w-full px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-500 disabled:opacity-50 text-sm font-medium"
        >
          {saving ? 'Saving...' : 'Save'}
        </button>
      </div>
      <div className="flex-1 border rounded-lg overflow-hidden">
        <div className="bg-gray-50 px-3 py-2 border-b text-xs font-medium text-gray-500">Template (Markdown)</div>
        <div ref={editorRef} className="h-full overflow-auto" />
      </div>
    </div>
  )
}
