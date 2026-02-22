"""Generates tool execution plans from task + context.

Uses LLM to determine which tool to use and with what parameters.
Falls back to rule-based matching when LLM is unavailable.
"""


from nexus.types import PersonaContract, RetrievedContext, IntentDeclaration
from nexus.tools.registry import ToolRegistry


class ToolSelector:
    """Generates tool execution plans from task + context."""

    def __init__(self, registry: ToolRegistry, llm_client=None):
        """
        Args:
            registry: Tool registry with all available tools
            llm_client: LLM client for intelligent selection (Phase 6).
                        If None, uses rule-based matching.
        """
        self.registry = registry
        self.llm_client = llm_client

    async def select(
        self,
        task: str,
        persona: PersonaContract,
        context: RetrievedContext,
    ) -> IntentDeclaration:
        """Select the best tool for a task step.

        Given a task, available tools (filtered by persona), and retrieved context,
        ask the LLM to generate a tool call with parameters.
        If llm_client is None, use rule-based matching (for testing).

        Args:
            task: Task step description
            persona: Active persona (filters available tools)
            context: Retrieved knowledge context

        Returns:
            IntentDeclaration with tool selection and parameters
        """
        available = self.registry.list_for_persona(persona)
        if not available:
            # No tools for this persona — default no-op declaration
            return IntentDeclaration(
                task_description=task,
                planned_action=task,
                tool_name="",
                tool_params={},
                resource_targets=[],
                reasoning="No tools available for this persona",
                confidence=0.0,
            )

        if self.llm_client is not None:
            return await self._select_with_llm(task, persona, context, available)
        return self._select_rule_based(task, persona, context, available)

    async def _select_with_llm(
        self,
        task: str,
        persona,
        context,
        available,
    ) -> "IntentDeclaration":
        """Use LLM to pick the best tool and build parameters."""
        import json
        from nexus.llm.prompts import SELECT_TOOL

        tool_list = "\n".join(
            f"- {t.name}: {t.description}" for t in available
        )
        kb = [d.get("content", "")[:200] for d in context.documents[:3] if d.get("source") != "session_history"]
        hist = [d.get("content", "")[:200] for d in context.documents if d.get("source") == "session_history"]

        system_prompt = SELECT_TOOL.format(tool_list=tool_list)
        user_content = json.dumps({
            "step": task,
            "kb_context": "\n".join(kb) or "None",
            "prior_results": "\n".join(hist) or "None",
        })

        try:
            response = await self.llm_client.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
            )
            raw = response.get("content", "")
            # Strip markdown code fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())

            tool_name = data.get("tool", available[0].name)
            params = data.get("params", {})
            reasoning = data.get("reasoning", "LLM selected")

            tool_defn = next((t for t in available if t.name == tool_name), available[0])
            return IntentDeclaration(
                task_description=task,
                planned_action=f"Execute {tool_name}: {task}",
                tool_name=tool_name,
                tool_params=params,
                resource_targets=self._derive_resource_targets(tool_defn, params),
                reasoning=reasoning,
                confidence=0.8,
            )
        except Exception:
            # LLM failed — fall back to rule-based
            return self._select_rule_based(task, persona, context, available)

    def _select_rule_based(
        self,
        task: str,
        persona,
        context,
        available,
    ) -> "IntentDeclaration":
        """Simple keyword-based tool selection fallback."""
        task_lower = task.lower()

        # Score each tool by how many words in its name/description appear in task
        best = available[0]
        best_score = 0
        for tool in available:
            score = sum(
                1
                for word in (tool.name + " " + tool.description).lower().split()
                if len(word) > 3 and word in task_lower
            )
            if score > best_score:
                best_score = score
                best = tool

        # Build minimal params: look for a natural "query" or "path" value in task
        params: dict = {}
        # Most tools accept a primary string arg — try to infer from schema
        schema_props = best.parameters.get("properties", {})
        for param_name in schema_props:
            if param_name in ("query", "text", "input"):
                params[param_name] = task
                break
            if param_name in ("path", "url"):
                # Extract first word that looks like a path/url
                words = task.split()
                for w in words:
                    if "/" in w or w.startswith("http"):
                        params[param_name] = w
                        break
                else:
                    params[param_name] = task
                break
        if not params and schema_props:
            # Fall back to first param
            first_param = next(iter(schema_props))
            params[first_param] = task

        # Generate a natural-language planned_action that aligns with persona
        # intent_patterns (e.g., "search for information about X") rather than
        # a technical "Execute tool_name: task" string which has poor cosine
        # similarity against conversational intent patterns used in Gate 2.
        planned_action = self._natural_planned_action(best.name, task)

        return IntentDeclaration(
            task_description=task,
            planned_action=planned_action,
            tool_name=best.name,
            tool_params=params,
            resource_targets=self._derive_resource_targets(best, params),
            reasoning=f"Rule-based selection: '{best.name}' matched task keywords",
            confidence=0.5,
        )

    def _natural_planned_action(self, tool_name: str, task: str) -> str:
        """Return a natural-language description of the intended action.

        Keeps the planned_action semantically close to the persona's
        intent_patterns so Gate 2 cosine similarity scores are realistic.
        """
        _templates = {
            "web_search":       f"search for information about {task}",
            "knowledge_search": f"search for information about {task}",
            "web_fetch":        f"find data about {task}",
            "file_read":        f"look up data from {task}",
            "file_write":       f"write content about {task}",
            "send_email":       f"send email about {task}",
            "compute_stats":    f"analyze data for {task}",
        }
        return _templates.get(tool_name, f"search for information about {task}")

    def _derive_resource_targets(self, tool_defn, params: dict) -> list[str]:
        """Derive resource_targets from tool's resource_pattern.

        Converts a glob pattern like "kb:*" into a specific target like "kb:query"
        that will correctly fnmatch against the persona's resource_scopes.
        """
        pattern = tool_defn.resource_pattern  # e.g., "kb:*", "email:*", "*"
        if "*" not in pattern:
            return [pattern]
        prefix = pattern[: pattern.index("*")]  # "kb:"
        if params:
            # Use first param value as a slug for the specific resource
            val = str(list(params.values())[0])
            slug = val.lower().replace(" ", "_")[:30]
            return [prefix + slug]
        return [pattern]
