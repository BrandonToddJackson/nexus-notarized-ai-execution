"""Persona lifecycle management: load, activate, validate, revoke.

A persona is NOT a separate agent — it's a constrained operating mode
with allowed tools, resource scopes, and intent patterns. Ephemeral identity.

Implementation notes:
- Store active personas in a dict with activation timestamps
- Resource scope matching: glob patterns (e.g., `db:analytics.*` matches `db:analytics.customers`)
- TTL is checked in the anomaly gate, not here — but PersonaManager tracks the activation time
"""

import fnmatch
from datetime import datetime
from typing import Optional

from nexus.types import PersonaContract
from nexus.exceptions import PersonaViolation


class PersonaManager:
    """Manages persona lifecycle: load, activate, validate, revoke."""

    def __init__(self, personas: list[PersonaContract] = None):
        """Load persona contracts (from YAML or DB).

        Args:
            personas: List of persona contracts to manage. Can be empty at init
                      and loaded later via load_personas().
        """
        self._contracts: dict[str, PersonaContract] = {}
        self._active: dict[str, datetime] = {}  # persona_id -> activation timestamp
        if personas:
            for p in personas:
                self._contracts[p.name] = p

    def load_personas(self, personas: list[PersonaContract]) -> None:
        """Load or reload persona contracts."""
        for p in personas:
            self._contracts[p.name] = p

    def activate(self, persona_name: str, tenant_id: str) -> PersonaContract:
        """Activate a persona for use. Returns the contract.

        Sets activation timestamp for TTL tracking.

        Args:
            persona_name: Name of the persona to activate (e.g., "researcher")
            tenant_id: Tenant context for scoping

        Returns:
            The PersonaContract for this persona

        Raises:
            PersonaViolation: if persona doesn't exist or is disabled
        """
        persona = self._contracts.get(persona_name)
        if persona is None:
            raise PersonaViolation(f"Persona '{persona_name}' not found")
        if not persona.is_active:
            raise PersonaViolation(f"Persona '{persona_name}' is disabled")
        self._active[persona_name] = datetime.utcnow()
        return persona

    def validate_action(self, persona: PersonaContract, tool_name: str, resource_targets: list[str]) -> bool:
        """Check if action is within persona's behavioral contract.

        Checks:
        - tool_name must be in persona.allowed_tools
        - each resource_target must match at least one persona.resource_scopes pattern
          using fnmatch glob matching (e.g., "db:analytics.*" matches "db:analytics.customers")

        Args:
            persona: The active persona contract
            tool_name: Tool being requested
            resource_targets: Resources the tool will access

        Returns:
            True if valid

        Raises:
            PersonaViolation: with specific reason if invalid
        """
        if tool_name not in persona.allowed_tools:
            raise PersonaViolation(
                f"Tool '{tool_name}' not in persona '{persona.name}' allowed_tools: {persona.allowed_tools}"
            )
        for target in resource_targets:
            if not any(fnmatch.fnmatch(target, scope) for scope in persona.resource_scopes):
                raise PersonaViolation(
                    f"Resource '{target}' not within persona '{persona.name}' scopes: {persona.resource_scopes}"
                )
        return True

    def revoke(self, persona_name: str) -> None:
        """Deactivate persona. Called after every action (ephemeral identity).

        Args:
            persona_name: Name of persona to revoke
        """
        self._active.pop(persona_name, None)

    def get_ttl_remaining(self, persona_name: str) -> int:
        """Seconds remaining before persona TTL expires.

        Args:
            persona_name: Name of the active persona

        Returns:
            Seconds remaining, or 0 if not active / expired
        """
        activation = self._active.get(persona_name)
        if activation is None:
            return 0
        persona = self._contracts.get(persona_name)
        if persona is None:
            return 0
        elapsed = (datetime.utcnow() - activation).total_seconds()
        remaining = persona.max_ttl_seconds - elapsed
        return max(0, int(remaining))

    def get_activation_time(self, persona_name: str) -> Optional[datetime]:
        """Get when a persona was activated. Used by Gate 3 (TTL check)."""
        return self._active.get(persona_name)

    def list_personas(self) -> list[PersonaContract]:
        """Return all loaded persona contracts."""
        return list(self._contracts.values())

    def get_persona(self, name: str) -> Optional[PersonaContract]:
        """Get a persona contract by name."""
        return self._contracts.get(name)
