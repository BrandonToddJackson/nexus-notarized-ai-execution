"""Tests for LLM model routing: is_local_model, select_model, api_base injection.

Covers:
  Gap 12 — is_local_model detection, select_model routing per task type
  Gap 20 — asyncio timeout raises NexusError
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from nexus.llm.client import (
    is_local_model,
    select_model,
    LLMClient,
    TASK_CODE,
    TASK_VISION,
    TASK_GENERAL,
)
from nexus.exceptions import NexusError


def _make_llm_response(content: str = "hello") -> MagicMock:
    """Build a minimal litellm response mock."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = []
    resp.usage.prompt_tokens = 5
    resp.usage.completion_tokens = 5
    return resp


class TestIsLocalModel:

    @pytest.mark.parametrize("model,expected", [
        ("ollama/qwen2.5-coder:7b", True),
        ("ollama_chat/mistral:7b", True),
        ("ollama/qwen2.5vl:7b", True),
        ("ollama/any-model", True),
        ("ollama_chat/any-model", True),
        ("anthropic/claude-sonnet-4-20250514", False),
        ("openai/gpt-4o", False),
        ("", False),
        # "ollama" alone (no slash) is NOT a valid prefix
        ("ollamaX/model", False),
    ])
    def test_is_local_model(self, model: str, expected: bool) -> None:
        assert is_local_model(model) == expected


class TestSelectModel:

    def test_cloud_model_returns_default_for_code(self):
        with patch("nexus.llm.client.config") as mock_cfg:
            mock_cfg.default_llm_model = "anthropic/claude-sonnet-4-20250514"
            mock_cfg.ollama_code_model = "ollama/code"
            mock_cfg.ollama_vision_model = "ollama/vision"
            assert select_model(TASK_CODE) == "anthropic/claude-sonnet-4-20250514"

    def test_cloud_model_returns_default_for_vision(self):
        with patch("nexus.llm.client.config") as mock_cfg:
            mock_cfg.default_llm_model = "anthropic/claude-sonnet-4-20250514"
            mock_cfg.ollama_code_model = "ollama/code"
            mock_cfg.ollama_vision_model = "ollama/vision"
            assert select_model(TASK_VISION) == "anthropic/claude-sonnet-4-20250514"

    def test_cloud_model_returns_default_for_general(self):
        with patch("nexus.llm.client.config") as mock_cfg:
            mock_cfg.default_llm_model = "anthropic/claude-sonnet-4-20250514"
            mock_cfg.ollama_code_model = "ollama/code"
            mock_cfg.ollama_vision_model = "ollama/vision"
            assert select_model(TASK_GENERAL) == "anthropic/claude-sonnet-4-20250514"

    def test_ollama_code_task_routes_to_code_model(self):
        with patch("nexus.llm.client.config") as mock_cfg:
            mock_cfg.default_llm_model = "ollama/qwen2.5-coder:7b"
            mock_cfg.ollama_code_model = "ollama/code-model:latest"
            mock_cfg.ollama_vision_model = "ollama/vision-model:latest"
            assert select_model(TASK_CODE) == "ollama/code-model:latest"

    def test_ollama_vision_task_routes_to_vision_model(self):
        with patch("nexus.llm.client.config") as mock_cfg:
            mock_cfg.default_llm_model = "ollama/qwen2.5-coder:7b"
            mock_cfg.ollama_code_model = "ollama/code-model:latest"
            mock_cfg.ollama_vision_model = "ollama/vision-model:latest"
            assert select_model(TASK_VISION) == "ollama/vision-model:latest"

    def test_ollama_general_routes_to_vision_model(self):
        """For Ollama, 'general' task uses the multimodal/vision model."""
        with patch("nexus.llm.client.config") as mock_cfg:
            mock_cfg.default_llm_model = "ollama/qwen2.5-coder:7b"
            mock_cfg.ollama_code_model = "ollama/code-model:latest"
            mock_cfg.ollama_vision_model = "ollama/vision-model:latest"
            assert select_model(TASK_GENERAL) == "ollama/vision-model:latest"


class TestLLMClientApiBase:

    async def test_complete_injects_api_base_for_ollama(self):
        """Ollama models must receive api_base in the litellm.acompletion kwargs."""
        with patch("nexus.llm.client.config") as mock_cfg:
            mock_cfg.ollama_base_url = "http://localhost:11434"
            mock_cfg.llm_temperature = 0.7
            mock_cfg.llm_max_tokens = 4096

            with patch(
                "nexus.llm.client.litellm.acompletion", new_callable=AsyncMock
            ) as mock_comp:
                mock_comp.return_value = _make_llm_response("hello")

                client = LLMClient(model="ollama/qwen2.5-coder:7b")
                await client.complete([{"role": "user", "content": "hi"}])

                call_kwargs = mock_comp.call_args.kwargs
                assert "api_base" in call_kwargs
                assert call_kwargs["api_base"] == "http://localhost:11434"

    async def test_complete_does_not_inject_api_base_for_cloud(self):
        """Cloud models must NOT receive api_base in kwargs."""
        with patch("nexus.llm.client.config") as mock_cfg:
            mock_cfg.llm_temperature = 0.7
            mock_cfg.llm_max_tokens = 4096

            with patch(
                "nexus.llm.client.litellm.acompletion", new_callable=AsyncMock
            ) as mock_comp:
                mock_comp.return_value = _make_llm_response("hello")

                client = LLMClient(model="anthropic/claude-sonnet-4-20250514")
                await client.complete([{"role": "user", "content": "hi"}])

                call_kwargs = mock_comp.call_args.kwargs
                assert "api_base" not in call_kwargs

    async def test_complete_returns_content_and_usage(self):
        """complete() normalises the litellm response into the expected dict shape."""
        with patch("nexus.llm.client.config") as mock_cfg:
            mock_cfg.llm_temperature = 0.7
            mock_cfg.llm_max_tokens = 4096

            with patch(
                "nexus.llm.client.litellm.acompletion", new_callable=AsyncMock
            ) as mock_comp:
                mock_comp.return_value = _make_llm_response("test response")

                client = LLMClient(model="anthropic/claude-sonnet-4-20250514")
                result = await client.complete([{"role": "user", "content": "hi"}])

                assert result["content"] == "test response"
                assert "usage" in result
                assert "input_tokens" in result["usage"]
                assert "output_tokens" in result["usage"]

    async def test_complete_passes_tools_when_provided(self):
        """tools kwarg must be forwarded to litellm when supplied."""
        with patch("nexus.llm.client.config") as mock_cfg:
            mock_cfg.llm_temperature = 0.7
            mock_cfg.llm_max_tokens = 4096

            with patch(
                "nexus.llm.client.litellm.acompletion", new_callable=AsyncMock
            ) as mock_comp:
                mock_comp.return_value = _make_llm_response()

                tools = [{"type": "function", "function": {"name": "search"}}]
                client = LLMClient(model="anthropic/claude-sonnet-4-20250514")
                await client.complete([{"role": "user", "content": "hi"}], tools=tools)

                call_kwargs = mock_comp.call_args.kwargs
                assert "tools" in call_kwargs
                assert call_kwargs["tools"] == tools


class TestLLMClientTimeout:

    async def test_complete_timeout_raises_nexus_error(self):
        """asyncio.wait_for timeout → NexusError with 'LLM call failed' message."""
        with patch("nexus.llm.client.config") as mock_cfg:
            mock_cfg.llm_temperature = 0.7
            mock_cfg.llm_max_tokens = 4096

            with patch(
                "nexus.llm.client.asyncio.wait_for",
                side_effect=asyncio.TimeoutError("timed out"),
            ):
                client = LLMClient(model="anthropic/claude-sonnet-4-20250514")
                with pytest.raises(NexusError, match="LLM call failed"):
                    await client.complete([{"role": "user", "content": "hi"}])

    async def test_complete_provider_error_raises_nexus_error(self):
        """Any provider exception → NexusError wrapper."""
        with patch("nexus.llm.client.config") as mock_cfg:
            mock_cfg.llm_temperature = 0.7
            mock_cfg.llm_max_tokens = 4096

            with patch(
                "nexus.llm.client.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=RuntimeError("connection refused"),
            ):
                client = LLMClient(model="anthropic/claude-sonnet-4-20250514")
                with pytest.raises(NexusError, match="LLM call failed"):
                    await client.complete([{"role": "user", "content": "hi"}])
