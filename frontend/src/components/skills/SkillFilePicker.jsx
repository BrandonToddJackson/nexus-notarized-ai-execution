import { useRef } from 'react'

const MAX_FILES = 5
const MAX_SIZE = 500 * 1024

export function SkillFilePicker({ files = [], onChange }) {
  const inputRef = useRef(null)

  function handleAdd(e) {
    const newFiles = Array.from(e.target.files || [])
    const valid = newFiles.filter(f => {
      if (f.size > MAX_SIZE) return false
      if (files.length + newFiles.indexOf(f) >= MAX_FILES) return false
      return true
    })
    onChange([...files, ...valid.map(f => ({ file: f, name: f.name, size: f.size, description: '' }))])
    if (inputRef.current) inputRef.current.value = ''
  }

  function handleRemove(index) {
    onChange(files.filter((_, i) => i !== index))
  }

  function handleDescChange(index, desc) {
    const next = [...files]
    next[index] = { ...next[index], description: desc }
    onChange(next)
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={files.length >= MAX_FILES}
          className="px-3 py-1.5 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
        >
          Add File
        </button>
        <span className="text-xs text-gray-500">{files.length}/{MAX_FILES} files (max 500KB each)</span>
      </div>
      <input ref={inputRef} type="file" onChange={handleAdd} className="hidden" multiple />
      {files.map((f, i) => (
        <div key={i} className="flex items-center gap-2 bg-gray-50 rounded p-2">
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-gray-700 truncate">{f.name}</div>
            <div className="text-xs text-gray-400">{(f.size / 1024).toFixed(1)} KB</div>
            <input
              type="text"
              value={f.description}
              onChange={e => handleDescChange(i, e.target.value)}
              placeholder="Description..."
              className="mt-1 w-full text-xs border border-gray-200 rounded px-2 py-1"
            />
          </div>
          <button onClick={() => handleRemove(i)} className="text-red-400 hover:text-red-600 text-sm px-2">x</button>
        </div>
      ))}
    </div>
  )
}
