export function EmptyState({ icon, title, description, actionLabel, onAction }) {
  return (
    <div className="text-center py-16">
      {icon && <div className="text-4xl mb-4">{icon}</div>}
      <h3 className="text-lg font-medium text-gray-900 mb-2">{title}</h3>
      {description && <p className="text-sm text-gray-500 mb-4">{description}</p>}
      {actionLabel && onAction && (
        <button
          onClick={onAction}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg"
        >
          {actionLabel}
        </button>
      )}
    </div>
  )
}
