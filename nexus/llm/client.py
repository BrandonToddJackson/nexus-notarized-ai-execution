"""Thin wrapper around litellm for NEXUS-specific usage.

litellm handles Anthropic, OpenAI, Ollama, and 100+ providers.
This wrapper adds: model routing, error normalization, usage extraction.

Model routing
─────────────
Cloud providers (Anthropic, OpenAI, etc.) handle all task types through one
model — routing is a no-op.

When using Ollama, the system automatically picks the right local model based
on what the task needs:

  task_type="code"    → config.ollama_code_model    (qwen2.5-coder:7b)
  task_type="vision"  → config.ollama_vision_model  (qwen2.5vl:7b)
  task_type="general" → config.ollama_vision_model  (qwen2.5vl:7b — best general purpose)

Usage:
  LLMClient(task_type="code")    # code gen, tool use, agentic tasks
  LLMClient(task_type="vision")  # image/chart/OCR analysis
  LLMClient(task_type="general") # default — text reasoning, planning
  LLMClient(model="ollama/...")  # explicit override, bypasses routing
"""

import asyncio

import litellm
from nexus.config import config
from nexus.exceptions import NexusError

# Task types understood by the router
TASK_CODE    = "code"
TASK_VISION  = "vision"
TASK_GENERAL = "general"


def is_local_model(model: str) -> bool:
    """Return True if the model runs locally via Ollama (no API key required)."""
    return model.startswith("ollama/") or model.startswith("ollama_chat/")


def select_model(task_type: str = TASK_GENERAL) -> str:
    """Return the best model for the given task type.

    For cloud providers, the configured default handles everything.
    For Ollama, routes to the specialised local model for the task.

    Args:
        task_type: "code", "vision", or "general" (default).

    Returns:
        Model string suitable for litellm (e.g. "ollama/qwen2.5-coder:7b").
    """
    base = config.default_llm_model
    if not is_local_model(base):
        # Cloud model — handles all task types natively
        return base
    if task_type == TASK_CODE:
        return config.ollama_code_model
    # vision and general both go to the multimodal model
    return config.ollama_vision_model


class LLMClient:
    """Thin wrapper around litellm for NEXUS-specific usage."""

    def __init__(self, model: str = None, task_type: str = None):
        """
        Args:
            model:     Explicit model string — bypasses routing.
                         e.g. "anthropic/claude-sonnet-4-20250514", "ollama/qwen2.5vl:7b"
            task_type: Hint for automatic model selection when using Ollama:
                         "code"    — code generation, tool use, agentic reasoning
                         "vision"  — images, charts, OCR, multimodal
                         "general" — text reasoning, planning (default)
                       Ignored when an explicit model is passed or when using a
                       cloud provider (cloud handles all types via one model).
        """
        if model:
            self.model = model
        else:
            self.model = select_model(task_type or TASK_GENERAL)
        litellm.drop_params = True  # ignore unsupported params per provider

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        temperature: float = None,
        max_tokens: int = None,
        response_format: dict = None,
    ) -> dict:
        """Call LLM via litellm.acompletion().

        Args:
            messages:        Chat messages [{"role": "user", "content": "..."}]
            tools:           Optional tool definitions for function calling
            temperature:     Override config temperature
            max_tokens:      Override config max_tokens
            response_format: Optional response format constraint (e.g. JSON mode)

        Returns:
            {"content": str, "tool_calls": list, "usage": {"input_tokens": int, "output_tokens": int}}

        Raises:
            NexusError: On any LLM provider error
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": config.llm_temperature if temperature is None else temperature,
                "max_tokens": config.llm_max_tokens if max_tokens is None else max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
            if response_format:
                kwargs["response_format"] = response_format
            if is_local_model(self.model):
                kwargs["api_base"] = config.ollama_base_url

            response = await asyncio.wait_for(litellm.acompletion(**kwargs), timeout=30)

            choice = response.choices[0]
            return {
                "content": choice.message.content or "",
                "tool_calls": choice.message.tool_calls or [],
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }
            }
        except Exception as e:
            raise NexusError(f"LLM call failed: {e}", details={"model": self.model})
