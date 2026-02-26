import { useState } from 'react'

export function ConfirmDelete({ title, message, confirmText, onConfirm, onCancel }) {
  const [typed, setTyped] = useState('')
  const needsTyping = !!confirmText
  const canConfirm = !needsTyping || typed === confirmText

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onCancel}>
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 p-6" onClick={e => e.stopPropagation()}>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">{title || 'Confirm Delete'}</h3>
        <p className="text-sm text-gray-600 mb-4">{message || 'This action cannot be undone.'}</p>
        {needsTyping && (
          <div className="mb-4">
            <label className="block text-sm text-gray-700 mb-1">
              Type <span className="font-mono font-bold">{confirmText}</span> to confirm:
            </label>
            <input
              type="text"
              value={typed}
              onChange={e => setTyped(e.target.value)}
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-red-500"
            />
          </div>
        )}
        <div className="flex justify-end gap-3">
          <button onClick={onCancel} className="px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded">
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={!canConfirm}
            className="px-4 py-2 text-sm text-white bg-red-600 hover:bg-red-700 rounded disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  )
}
