import { create } from 'zustand'

export const useAppStore = create((set) => ({
  sidebarCollapsed: false,
  toggleSidebar: () => set(s => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  selectedIds: new Set(),
  toggleSelected: (id) => set(s => {
    const next = new Set(s.selectedIds)
    next.has(id) ? next.delete(id) : next.add(id)
    return { selectedIds: next }
  }),
  clearSelected: () => set({ selectedIds: new Set() }),
  gateFailureCount: 0,
  incrementGateFailures: () => set(s => ({ gateFailureCount: s.gateFailureCount + 1 })),
  clearGateFailures: () => set({ gateFailureCount: 0 }),
}))
