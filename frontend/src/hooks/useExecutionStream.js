import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '../stores/appStore.js'
import { getToken } from '../lib/auth.js'
import toast from 'react-hot-toast'

export const useExecutionStream = () => {
  const qc = useQueryClient()
  const incrementGateFailures = useAppStore(s => s.incrementGateFailures)
  useEffect(() => {
    const token = getToken()
    if (!token) return
    const url = `/v2/events/stream?token=${encodeURIComponent(token)}`
    const es = new EventSource(url)
    es.addEventListener('execution_update', () => {
      qc.invalidateQueries({ queryKey: ['executions'] })
    })
    es.addEventListener('gate_failure', () => {
      incrementGateFailures()
      toast.error('Gate failure detected', { duration: 4000 })
    })
    return () => es.close()
  }, [qc, incrementGateFailures])
}
