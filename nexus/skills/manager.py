"""SkillManager — lifecycle management for SkillRecord objects.

Supports both in-memory operation (no repository, for tests and CLI) and
full persistence when a Repository is provided, matching the pattern used
by WorkflowManager.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from nexus.config import NexusConfig
from nexus.exceptions import SkillNotFound, SkillValidationError
from nexus.types import SkillRecord, SkillVersion, SkillInvocation, SkillFile

# Claude Code tool name → NEXUS tool name mapping
_CLAUDE_TOOL_MAP: dict[str, str] = {
    "Read": "file_reader_tool",
    "Bash": "code_execute_python",
    "Glob": "file_reader_tool",
    "Grep": "knowledge_search",
    "Edit": "file_reader_tool",
    "Write": "file_reader_tool",
    "WebFetch": "http_request",
    "WebSearch": "knowledge_search",
}


class SkillManager:
    """Manages the full lifecycle of SkillRecord objects.

    Args:
        repository:        Optional Repository instance for persistence.
        embedding_service: Optional EmbeddingService for semantic matching.
        config:            NexusConfig instance.
    """

    def __init__(
        self,
        repository: Any = None,
        embedding_service: Any = None,
        config: Optional[NexusConfig] = None,
    ) -> None:
        self._repository = repository
        self._embedding_service = embedding_service
        self._config = config or NexusConfig()
        self._store: dict[str, SkillRecord] = {}

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_name(self, name: str) -> None:
        if not name or len(name) > 64:
            raise SkillValidationError("Skill name must be 1-64 characters")
        if not re.match(r"^[a-z0-9][a-z0-9-]*$", name):
            raise SkillValidationError(
                "Skill name must be lowercase alphanumeric with hyphens, starting with alphanumeric"
            )

    def _validate(self, skill: SkillRecord) -> None:
        self._validate_name(skill.name)
        if len(skill.description) > 1024:
            raise SkillValidationError("Skill description must be at most 1024 characters")
        if not skill.display_name:
            raise SkillValidationError("Skill display_name is required")
        if not skill.content:
            raise SkillValidationError("Skill content is required")

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def create(
        self,
        tenant_id: str,
        name: str,
        display_name: str,
        description: str,
        content: str,
        allowed_tools: Optional[list[str]] = None,
        allowed_personas: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        supporting_files: Optional[list[SkillFile]] = None,
    ) -> SkillRecord:
        now = datetime.now(tz=timezone.utc)
        skill = SkillRecord(
            id=str(uuid4()),
            tenant_id=tenant_id,
            name=name,
            display_name=display_name,
            description=description,
            content=content,
            allowed_tools=allowed_tools or [],
            allowed_personas=allowed_personas or [],
            tags=tags or [],
            supporting_files=supporting_files or [],
            created_at=now,
            updated_at=now,
        )
        self._validate(skill)
        self._store[skill.id] = skill
        if self._repository:
            try:
                await self._repository.create_skill(skill)
            except Exception:
                pass
        return skill

    async def get(self, skill_id: str, tenant_id: str) -> SkillRecord:
        skill = self._store.get(skill_id)
        if skill is None and self._repository:
            try:
                skill = await self._repository.get_skill(skill_id, tenant_id)
                if skill:
                    self._store[skill.id] = skill
            except Exception:
                pass
        if skill is None or skill.tenant_id != tenant_id:
            raise SkillNotFound(f"Skill '{skill_id}' not found")
        return skill

    async def update(
        self,
        skill_id: str,
        tenant_id: str,
        change_note: str,
        name: Optional[str] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        content: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
        allowed_personas: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        supporting_files: Optional[list[SkillFile]] = None,
        active: Optional[bool] = None,
    ) -> SkillRecord:
        existing = await self.get(skill_id, tenant_id)

        # Snapshot current version into history
        version_entry = SkillVersion(
            version=existing.version,
            content=existing.content,
            description=existing.description,
            changed_at=existing.updated_at,
            change_note=change_note,
        )

        updates: dict[str, Any] = {
            "updated_at": datetime.now(tz=timezone.utc),
            "version": existing.version + 1,
            "version_history": list(existing.version_history) + [version_entry],
        }
        if name is not None:
            updates["name"] = name
        if display_name is not None:
            updates["display_name"] = display_name
        if description is not None:
            updates["description"] = description
        if content is not None:
            updates["content"] = content
        if allowed_tools is not None:
            updates["allowed_tools"] = allowed_tools
        if allowed_personas is not None:
            updates["allowed_personas"] = allowed_personas
        if tags is not None:
            updates["tags"] = tags
        if supporting_files is not None:
            updates["supporting_files"] = supporting_files
        if active is not None:
            updates["active"] = active

        updated = existing.model_copy(update=updates)
        if name is not None:
            self._validate_name(name)
        self._store[skill_id] = updated

        if self._repository:
            try:
                repo_updates = dict(updates)
                if "version_history" in repo_updates:
                    repo_updates["version_history"] = [
                        v.model_dump(mode="json") for v in repo_updates["version_history"]
                    ]
                if "supporting_files" in repo_updates:
                    repo_updates["supporting_files"] = [
                        f.model_dump() for f in repo_updates["supporting_files"]
                    ]
                await self._repository.update_skill(skill_id, repo_updates)
            except Exception:
                pass

        return updated

    async def list(
        self,
        tenant_id: str,
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SkillRecord]:
        results = [
            s for s in self._store.values()
            if s.tenant_id == tenant_id
            and (not active_only or s.active)
        ]
        results.sort(key=lambda s: s.created_at, reverse=True)
        return results[offset: offset + limit]

    async def delete(self, skill_id: str, tenant_id: str) -> bool:
        skill = self._store.get(skill_id)
        if skill is None or skill.tenant_id != tenant_id:
            raise SkillNotFound(f"Skill '{skill_id}' not found")
        updated = skill.model_copy(update={"active": False, "updated_at": datetime.now(tz=timezone.utc)})
        self._store[skill_id] = updated
        if self._repository:
            try:
                await self._repository.delete_skill(skill_id, tenant_id)
            except Exception:
                pass
        return True

    # ── Duplicate ─────────────────────────────────────────────────────────────

    async def duplicate(self, skill_id: str, tenant_id: str) -> SkillRecord:
        source = await self.get(skill_id, tenant_id)
        return await self.create(
            tenant_id=tenant_id,
            name=f"{source.name}-copy",
            display_name=f"{source.display_name} (copy)",
            description=source.description,
            content=source.content,
            allowed_tools=list(source.allowed_tools),
            allowed_personas=list(source.allowed_personas),
            tags=list(source.tags),
            supporting_files=list(source.supporting_files),
        )

    # ── Export / Import ───────────────────────────────────────────────────────

    def export_json(self, skill: SkillRecord) -> str:
        return skill.model_dump_json(indent=2)

    async def import_json(self, raw: str, tenant_id: str) -> SkillRecord:
        """Import from JSON string or SKILL.md with YAML frontmatter."""
        raw = raw.strip()

        if raw.startswith("---"):
            return await self._import_frontmatter(raw, tenant_id)

        data = json.loads(raw)
        # Map Claude Code tool names to NEXUS names
        allowed_tools = [
            _CLAUDE_TOOL_MAP.get(t, t) for t in data.get("allowed_tools", [])
        ]
        return await self.create(
            tenant_id=tenant_id,
            name=data.get("name", f"imported-{str(uuid4())[:8]}"),
            display_name=data.get("display_name", data.get("name", "Imported Skill")),
            description=data.get("description", ""),
            content=data.get("content", ""),
            allowed_tools=allowed_tools,
            allowed_personas=data.get("allowed_personas", []),
            tags=data.get("tags", ["imported"]),
        )

    async def _import_frontmatter(self, raw: str, tenant_id: str) -> SkillRecord:
        """Parse SKILL.md with YAML frontmatter."""
        import yaml
        parts = raw.split("---", 2)
        if len(parts) < 3:
            raise SkillValidationError("Invalid YAML frontmatter format")
        meta = yaml.safe_load(parts[1]) or {}
        content = parts[2].strip()

        allowed_tools = [
            _CLAUDE_TOOL_MAP.get(t, t) for t in meta.get("allowed_tools", [])
        ]
        return await self.create(
            tenant_id=tenant_id,
            name=meta.get("name", f"imported-{str(uuid4())[:8]}"),
            display_name=meta.get("display_name", meta.get("name", "Imported Skill")),
            description=meta.get("description", ""),
            content=content,
            allowed_tools=allowed_tools,
            allowed_personas=meta.get("allowed_personas", []),
            tags=meta.get("tags", ["imported"]),
        )

    # ── Invocations ───────────────────────────────────────────────────────────

    async def record_invocation(
        self,
        skill_id: str,
        tenant_id: str,
        persona_name: str,
        execution_id: Optional[str] = None,
        workflow_name: Optional[str] = None,
        context_summary: str = "",
    ) -> SkillInvocation:
        skill = await self.get(skill_id, tenant_id)
        now = datetime.now(tz=timezone.utc)
        inv = SkillInvocation(
            id=str(uuid4()),
            skill_id=skill_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            workflow_name=workflow_name,
            persona_name=persona_name,
            context_summary=context_summary,
            invoked_at=now,
        )
        # Update invocation count in-memory
        updated = skill.model_copy(update={
            "invocation_count": skill.invocation_count + 1,
            "last_invoked_at": now,
        })
        self._store[skill_id] = updated

        if self._repository:
            try:
                await self._repository.record_skill_invocation(inv)
                await self._repository.update_skill(skill_id, {
                    "invocation_count": updated.invocation_count,
                    "last_invoked_at": now,
                })
            except Exception:
                pass
        return inv

    # ── Persona filtering ─────────────────────────────────────────────────────

    async def get_active_for_persona(
        self, tenant_id: str, persona_name: str
    ) -> list[SkillRecord]:
        all_skills = await self.list(tenant_id, active_only=True)
        return [
            s for s in all_skills
            if not s.allowed_personas or persona_name in s.allowed_personas
        ]

    # ── Semantic matching ─────────────────────────────────────────────────────

    async def semantic_match(
        self,
        query: str,
        tenant_id: str,
        persona_name: Optional[str] = None,
        top_k: int = 3,
    ) -> list[SkillRecord]:
        candidates = await self.get_active_for_persona(tenant_id, persona_name or "") \
            if persona_name else await self.list(tenant_id, active_only=True)

        if not candidates:
            return []

        if self._embedding_service:
            try:
                query_emb = await self._embedding_service.embed(query)
                scored = []
                for skill in candidates:
                    skill_text = f"{skill.name} {skill.description} {skill.content[:200]}"
                    skill_emb = await self._embedding_service.embed(skill_text)
                    similarity = sum(a * b for a, b in zip(query_emb, skill_emb))
                    scored.append((similarity, skill))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [s for _, s in scored[:top_k]]
            except Exception:
                pass

        # Fallback: substring match on description + content
        query_lower = query.lower()
        scored = []
        for skill in candidates:
            text = f"{skill.name} {skill.description} {skill.content}".lower()
            score = sum(1 for word in query_lower.split() if word in text)
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]
