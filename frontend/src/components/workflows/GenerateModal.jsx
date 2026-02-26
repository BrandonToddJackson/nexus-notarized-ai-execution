/**
 * GenerateModal — wraps NLGenerateModal (Phase 24) for use in the Workflows page.
 * Reuses the existing NL generation + ambiguity flow.
 */
import NLGenerateModal from '../workflow/modals/NLGenerateModal'

export function GenerateModal({ onGenerated, onClose }) {
  return <NLGenerateModal onGenerated={onGenerated} onClose={onClose} />
}
