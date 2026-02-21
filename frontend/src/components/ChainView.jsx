import React from 'react'
import SealCard from './SealCard'

const STATUS_CONNECTOR = {
  executed: 'bg-green-500',
  blocked: 'bg-red-500',
  failed: 'bg-orange-500',
  pending: 'bg-gray-600',
}

export default function ChainView({ seals = [], currentStep = -1 }) {
  if (seals.length === 0 && currentStep === -1) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-8 text-center">
        <p className="text-gray-400">No steps yet. Submit a task to begin.</p>
      </div>
    )
  }

  return (
    <div className="relative">
      {/* Vertical timeline line */}
      <div className="absolute left-3 top-3 bottom-3 w-0.5 bg-gray-700" />

      <div className="space-y-4">
        {seals.map((seal, i) => {
          const isCurrent = i === currentStep
          const status = seal.status || 'pending'
          const connectorColor = STATUS_CONNECTOR[status] || STATUS_CONNECTOR.pending

          return (
            <div key={seal.seal_id || seal.id || i} className="relative flex items-start gap-4">
              {/* Timeline connector circle */}
              <div className="relative z-10 flex-shrink-0 mt-4">
                {isCurrent ? (
                  <div className="w-6 h-6 rounded-full border-2 border-blue-400 flex items-center justify-center">
                    <div className="w-3 h-3 rounded-full bg-blue-400 animate-pulse" />
                  </div>
                ) : (
                  <div className={`w-6 h-6 rounded-full ${connectorColor}`} />
                )}
              </div>

              {/* Seal card */}
              <div className="flex-1 min-w-0">
                <SealCard seal={seal} />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
