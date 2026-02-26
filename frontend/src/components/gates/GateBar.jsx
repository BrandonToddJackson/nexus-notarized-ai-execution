import { GateChip } from './GateChip.jsx'

export function GateBar({ gates = [], compact = false }) {
  if (!gates?.length) return null
  if (compact) return (
    <div className="flex gap-1">
      {gates.map(g => <GateChip key={g.gate ?? g.name} {...g} />)}
    </div>
  )
  return (
    <div className="flex flex-wrap gap-2">
      {gates.map(g => <GateChip key={g.gate ?? g.name} {...g} />)}
    </div>
  )
}
