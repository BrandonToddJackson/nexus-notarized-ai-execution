import { useState } from 'react'
import { testCredential } from '../../api/credentials.js'

export function CredentialForm({ type, onSubmit, onCancel }) {
  const [form, setForm] = useState({ name: '', type: type?.type || 'custom', data: {} })
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)

  const fields = type?.fields || [
    { name: 'api_key', label: 'API Key', type: 'password' },
  ]

  function handleFieldChange(name, value) {
    setForm(f => ({ ...f, data: { ...f.data, [name]: value } }))
  }

  async function handleTest() {
    setTesting(true)
    try {
      const result = await testCredential(form)
      setTestResult(result)
    } catch (e) {
      setTestResult({ success: false, error: e.message })
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
        <input
          value={form.name}
          onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
          className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
          placeholder="My API Key"
        />
      </div>
      {fields.map(field => (
        <div key={field.name}>
          <label className="block text-sm font-medium text-gray-700 mb-1">{field.label || field.name}</label>
          <input
            type={field.type === 'password' ? 'password' : 'text'}
            value={form.data[field.name] || ''}
            onChange={e => handleFieldChange(field.name, e.target.value)}
            className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
          />
        </div>
      ))}
      {testResult && (
        <div className={`text-sm p-2 rounded ${testResult.success ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
          {testResult.success ? 'Connection successful' : `Failed: ${testResult.error || 'unknown error'}`}
        </div>
      )}
      <div className="flex gap-2">
        <button onClick={handleTest} disabled={testing} className="px-3 py-2 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50">
          {testing ? 'Testing...' : 'Test Connection'}
        </button>
        <div className="flex-1" />
        <button onClick={onCancel} className="px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded">Cancel</button>
        <button onClick={() => onSubmit(form)} className="px-3 py-2 text-sm bg-indigo-600 text-white rounded hover:bg-indigo-500">Save</button>
      </div>
    </div>
  )
}
