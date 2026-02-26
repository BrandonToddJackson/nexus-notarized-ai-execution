import { createPatch } from 'diff'

export function SkillVersionDiff({ oldText, newText, oldVersion, newVersion }) {
  const patch = createPatch(
    'skill',
    oldText || '',
    newText || '',
    `v${oldVersion || '?'}`,
    `v${newVersion || '?'}`
  )
  const lines = patch.split('\n')

  return (
    <pre className="text-xs font-mono bg-gray-50 rounded p-4 overflow-x-auto">
      {lines.map((line, i) => {
        let color = 'text-gray-600'
        if (line.startsWith('+') && !line.startsWith('+++')) color = 'text-green-700 bg-green-50'
        else if (line.startsWith('-') && !line.startsWith('---')) color = 'text-red-700 bg-red-50'
        else if (line.startsWith('@@')) color = 'text-blue-600'
        return <div key={i} className={color}>{line}</div>
      })}
    </pre>
  )
}
