import { useState } from 'react'

export function DataTable({ columns, data, onSort, page, totalPages, onPageChange, renderExpanded }) {
  const [expandedRow, setExpandedRow] = useState(null)

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead className="text-xs text-gray-500 uppercase bg-gray-50 border-b">
          <tr>
            {renderExpanded && <th className="px-4 py-3 w-8" />}
            {columns.map(col => (
              <th
                key={col.key}
                className={`px-4 py-3 ${onSort ? 'cursor-pointer hover:text-gray-700' : ''}`}
                onClick={() => onSort?.(col.key)}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {data?.map((row, i) => (
            <>
              <tr key={row.id ?? i} className="hover:bg-gray-50">
                {renderExpanded && (
                  <td className="px-4 py-3">
                    <button
                      onClick={() => setExpandedRow(expandedRow === i ? null : i)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      {expandedRow === i ? '\u25BC' : '\u25B6'}
                    </button>
                  </td>
                )}
                {columns.map(col => (
                  <td key={col.key} className="px-4 py-3">
                    {col.render ? col.render(row[col.key], row) : row[col.key]}
                  </td>
                ))}
              </tr>
              {renderExpanded && expandedRow === i && (
                <tr key={`${row.id ?? i}-expanded`}>
                  <td colSpan={columns.length + 1} className="px-4 py-3 bg-gray-50">
                    {renderExpanded(row)}
                  </td>
                </tr>
              )}
            </>
          ))}
        </tbody>
      </table>
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-3 border-t">
          <button
            onClick={() => onPageChange?.(page - 1)}
            disabled={page <= 1}
            className="text-sm text-gray-600 hover:text-gray-900 disabled:opacity-50"
          >
            Previous
          </button>
          <span className="text-sm text-gray-500">Page {page} of {totalPages}</span>
          <button
            onClick={() => onPageChange?.(page + 1)}
            disabled={page >= totalPages}
            className="text-sm text-gray-600 hover:text-gray-900 disabled:opacity-50"
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}
