"""Pydantic schemas for Plugin Marketplace API responses."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PluginToolResponse(BaseModel):
    name: str
    description: str
    risk_level: str
    allowed_personas: list[str]


class PluginManifestResponse(BaseModel):
    name: str
    version: str
    description: str
    author: str
    tools: list[PluginToolResponse]
    personas: list[str]
    nexus_version: str
    homepage: str
    license: str
    tags: list[str]
    verified: bool
    installed_at: Optional[str] = None
    checksum_sha256: str


class PluginSearchResult(BaseModel):
    name: str                # nexus-plugin-weather
    plugin_name: str         # weather
    version: str
    description: str
    author: str
    homepage: str
    downloads_last_month: int
    installed: bool
    installed_version: str
    verified: bool


class PluginInstallRequest(BaseModel):
    package_name: str
    version: Optional[str] = None
    force: bool = False


class PluginUpgradeRequest(BaseModel):
    version: Optional[str] = None
