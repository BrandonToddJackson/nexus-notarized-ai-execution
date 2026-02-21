"""Built-in tools package. Import to register all built-in tools."""

from nexus.tools.builtin.web import web_search, web_fetch
from nexus.tools.builtin.files import file_read, file_write
from nexus.tools.builtin.comms import send_email
from nexus.tools.builtin.data import compute_stats, knowledge_search

__all__ = [
    "web_search", "web_fetch",
    "file_read", "file_write",
    "send_email",
    "compute_stats", "knowledge_search",
]
