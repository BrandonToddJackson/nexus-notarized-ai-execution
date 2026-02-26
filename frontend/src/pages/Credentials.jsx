import { useState } from 'react'
import { useCredentials, useCreateCredential, useDeleteCredential } from '../hooks/useCredentials.js'
import { CredentialForm } from '../components/credentials/CredentialForm.jsx'
import { ConfirmDelete } from '../components/shared/ConfirmDelete.jsx'
import { DataTable } from '../components/shared/DataTable.jsx'
import { EmptyState } from '../components/shared/EmptyState.jsx'

export default function Credentials() {
  const { data: credentials, isLoading } = useCredentials()
  const createCredential = useCreateCredential()
  const deleteCredential = useDeleteCredential()
  const [showAdd, setShowAdd] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState(null)
  const credList = Array.isArray(credentials) ? credentials : credentials?.credentials || []

  const columns = [
    { key: 'name', label: 'Name' },
    { key: 'type', label: 'Type', render: (v) => <span className="text-xs bg-gray-100 px-2 py-0.5 rounded">{v}</span> },
    { key: 'created_at', label: 'Created', render: (v) => v ? new Date(v).toLocaleDateString() : '-' },
    { key: 'actions', label: '', render: (_, row) => (
      <button onClick={() => setDeleteTarget(row)} className="text-xs text-red-500 hover:text-red-700">Delete</button>
    )},
  ]

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Credentials</h1>
        <button
          onClick={() => setShowAdd(true)}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg"
        >
          Add Credential
        </button>
      </div>

      {isLoading ? (
        <div className="text-gray-500">Loading...</div>
      ) : credList.length === 0 ? (
        <EmptyState
          icon="\uD83D\uDD11"
          title="No credentials"
          description="Add credentials to connect to external services"
          actionLabel="Add Credential"
          onAction={() => setShowAdd(true)}
        />
      ) : (
        <div className="bg-white border rounded-lg">
          <DataTable columns={columns} data={credList} />
        </div>
      )}

      {showAdd && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setShowAdd(false)}>
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 p-6" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-semibold mb-4">Add Credential</h3>
            <CredentialForm
              onSubmit={(data) => {
                createCredential.mutate(data, { onSuccess: () => setShowAdd(false) })
              }}
              onCancel={() => setShowAdd(false)}
            />
          </div>
        </div>
      )}

      {deleteTarget && (
        <ConfirmDelete
          title="Delete Credential"
          message={`Are you sure you want to delete "${deleteTarget.name}"?`}
          onConfirm={() => {
            deleteCredential.mutate(deleteTarget.id)
            setDeleteTarget(null)
          }}
          onCancel={() => setDeleteTarget(null)}
        />
      )}
    </div>
  )
}
