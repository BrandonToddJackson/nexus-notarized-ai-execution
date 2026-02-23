"""Built-in tools package. Import to register all built-in tools."""

from nexus.tools.builtin.web import web_search, web_fetch
from nexus.tools.builtin.files import file_read, file_write
from nexus.tools.builtin.comms import send_email
from nexus.tools.builtin.data import compute_stats, knowledge_search
from nexus.tools.builtin.http_request import http_request
from nexus.tools.builtin.data_transform import data_transform
from nexus.tools.sandbox_v2 import code_execute_python, code_execute_javascript
from nexus.tools.builtin.frontend_design import generate_frontend_design

__all__ = [
    "web_search", "web_fetch",
    "file_read", "file_write",
    "send_email",
    "compute_stats", "knowledge_search",
    "http_request",
    "data_transform",
    "code_execute_python",
    "code_execute_javascript",
    "generate_frontend_design",
]
