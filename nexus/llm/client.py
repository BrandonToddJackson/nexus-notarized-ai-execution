"""Thin wrapper around litellm for NEXUS-specific usage.

litellm handles Anthropic, OpenAI, Ollama, and 100+ providers.
This wrapper adds: error normalization, usage extraction, NEXUS-specific defaults.
"""

import asyncio

import litellm
from nexus.config import config
from nexus.exceptions import NexusError


class LLMClient:
    """Thin wrapper around litellm for NEXUS-specific usage."""

    def __init__(self, model: str = None):
        """
        Args:
            model: LLM model string (e.g., "anthropic/claude-sonnet-4-20250514").
                   Defaults to config.default_llm_model.
        """
        self.model = model or config.default_llm_model
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
            messages: Chat messages [{"role": "user", "content": "..."}]
            tools: Optional tool definitions for function calling
            temperature: Override temperature
            max_tokens: Override max tokens
            response_format: Optional response format constraint

        Returns:
            {
                "content": str,
                "tool_calls": list,
                "usage": {"input_tokens": int, "output_tokens": int}
            }

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
