"""MCPToolAdapter — bridges MCPClient to the NEXUS ToolRegistry.

Responsibilities:
  - Connect to MCP servers and register their tools with source="mcp"
  - Unregister tools and disconnect when a server is removed
  - Persist server configs (in-memory; pass a repository for DB-backed storage)
  - Reconnect all previously registered servers on API startup
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from nexus.exceptions import MCPConnectionError
from nexus.mcp.client import MCPClient
from nexus.types import MCPServerConfig, ToolDefinition
from nexus.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from nexus.credentials.vault import CredentialVault

logger = logging.getLogger(__name__)


class MCPToolAdapter:
    """High-level bridge between MCP servers and the NEXUS tool registry.

    Args:
        registry:   NEXUS ToolRegistry to register/unregister tools in.
        client:     Optional pre-built MCPClient; one is created if not provided.
        repository: Optional async repository with ``save_mcp_server`` /
                    ``list_mcp_servers`` / ``delete_mcp_server`` methods.
                    When None, configs are stored in-memory only.
        vault:      Optional CredentialVault.  When an ``MCPServerConfig`` carries a
                    ``credential_id`` the vault decrypts the credential and injects
                    its values as subprocess environment variables before the server
                    process is spawned.  Credentials are never written to the ledger.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        client: MCPClient | None = None,
        repository: Any = None,
        vault: CredentialVault | None = None,
    ) -> None:
        self._registry = registry
        self._client = client or MCPClient()
        self._repository = repository
        self._vault = vault

        # server_id → MCPServerConfig
        self._servers: dict[str, MCPServerConfig] = {}
        # server_id → list of namespaced tool names registered in the registry
        self._server_tools: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Server management
    # ------------------------------------------------------------------

    async def register_server(
        self,
        tenant_id: str,
        server_config: MCPServerConfig,
    ) -> list[ToolDefinition]:
        """Connect to an MCP server and register its tools.

        All tools are registered with ``source="mcp"`` so they can be
        queried separately from built-in tools.

        Args:
            tenant_id: Owning tenant (for multi-tenancy scoping).
            server_config: Connection parameters for the MCP server.

        Returns:
            List of ToolDefinitions that were registered.

        Raises:
            MCPConnectionError: If the server cannot be reached.
        """
        server_config.tenant_id = tenant_id
        server_id = server_config.id

        # Inject vault credentials as subprocess env vars (if configured)
        if server_config.credential_id and self._vault is not None:
            server_config = self._inject_credential(server_config, tenant_id)

        definitions = await self._client.connect(server_config)

        # Register each tool in the NEXUS registry
        registered: list[str] = []
        for defn in definitions:
            # Build a closure capturing the correct server_id + tool name
            impl = self._make_tool_impl(server_id, defn.name)
            self._registry.register(defn, impl, source="mcp")
            registered.append(defn.name)

        self._servers[server_id] = server_config
        self._server_tools[server_id] = registered

        # Persist if repository is available
        if self._repository and hasattr(self._repository, "save_mcp_server"):
            try:
                await self._repository.save_mcp_server(server_config)
            except Exception:
                logger.warning("Failed to persist MCP server config for '%s'", server_config.name)

        logger.info(
            "Registered %d tools from MCP server '%s' (tenant=%s)",
            len(registered),
            server_config.name,
            tenant_id,
        )
        return definitions

    async def unregister_server(self, server_id: str) -> None:
        """Remove a server's tools from the registry and disconnect.

        Args:
            server_id: ID of the MCPServerConfig to remove.
        """
        tool_names = self._server_tools.pop(server_id, [])
        for name in tool_names:
            self._registry.unregister(name)

        self._servers.pop(server_id, None)
        await self._client.disconnect(server_id)

        if self._repository and hasattr(self._repository, "delete_mcp_server"):
            try:
                await self._repository.delete_mcp_server(server_id)
            except Exception:
                logger.warning("Failed to delete MCP server config '%s' from repo", server_id)

        logger.info("Unregistered MCP server '%s'", server_id)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_servers(self, tenant_id: str) -> list[MCPServerConfig]:
        """Return all registered servers for a tenant.

        Args:
            tenant_id: Tenant to filter by.

        Returns:
            List of MCPServerConfig objects.
        """
        return [cfg for cfg in self._servers.values() if cfg.tenant_id == tenant_id]

    def get_server(self, server_id: str) -> MCPServerConfig | None:
        """Return a server config by ID, or None if not found."""
        return self._servers.get(server_id)

    # ------------------------------------------------------------------
    # Startup reconnection
    # ------------------------------------------------------------------

    async def reconnect_all(self, tenant_id: str) -> None:
        """Reconnect all enabled servers for a tenant.

        Call this on API startup to restore MCP connections that were
        registered in a previous session (loaded from the repository).

        If the repository is available, loads configs from it first.
        Any server that fails to connect is logged but does not block others.

        Args:
            tenant_id: Tenant whose servers to reconnect.
        """
        configs: list[MCPServerConfig] = []

        if self._repository and hasattr(self._repository, "list_mcp_servers"):
            try:
                configs = await self._repository.list_mcp_servers(tenant_id)
            except Exception:
                logger.warning("Could not load MCP server configs from repository")
        else:
            # Fall back to in-memory list
            configs = self.list_servers(tenant_id)

        for cfg in configs:
            if not cfg.enabled:
                continue
            if self._client.is_connected(cfg.id):
                continue
            try:
                await self.register_server(tenant_id, cfg)
            except MCPConnectionError as exc:
                logger.warning(
                    "Failed to reconnect MCP server '%s': %s", cfg.name, exc
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_credential(
        self,
        server_config: MCPServerConfig,
        tenant_id: str,
    ) -> MCPServerConfig:
        """Return a copy of *server_config* with vault credentials merged into env.

        The vault credential is decrypted and its values are merged into the
        subprocess environment.  For MCP servers that need API keys (like Context7,
        Upstash, etc.) store a CUSTOM credential whose keys are the exact env var
        names the server expects, e.g.:

            vault.store(
                tenant_id="t1",
                name="context7-creds",
                credential_type=CredentialType.CUSTOM,
                service_name="context7",
                data={
                    "UPSTASH_REDIS_REST_URL": "https://...",
                    "UPSTASH_REDIS_REST_TOKEN": "AX...",
                },
            )

        Args:
            server_config: Original server config (not mutated).
            tenant_id:     Used for vault authorization check.

        Returns:
            A new MCPServerConfig with credential env vars merged in.
        """
        assert self._vault is not None  # guarded by caller
        try:
            env_vars = self._vault.get_env_vars(
                server_config.credential_id,  # type: ignore[arg-type]
                tenant_id,
            )
        except Exception as exc:
            logger.error(
                "[MCPAdapter] Failed to inject credential '%s' for server '%s': %s",
                server_config.credential_id,
                server_config.name,
                exc,
            )
            raise

        # Merge: explicit server env overrides credential env (intentional)
        merged_env = {**env_vars, **server_config.env}
        return server_config.model_copy(update={"env": merged_env})

    def _make_tool_impl(self, server_id: str, namespaced_name: str) -> Callable:
        """Return an async callable that routes execution to the MCP server."""

        async def _impl(**params: Any) -> Any:
            return await self._client.call_tool(server_id, namespaced_name, params)

        _impl.__name__ = namespaced_name
        return _impl
