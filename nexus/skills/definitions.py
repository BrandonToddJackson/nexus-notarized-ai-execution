"""Default skill definitions — named tool bundles with execution order."""

from nexus.types import SkillDefinition


DEFAULT_SKILLS = [
    SkillDefinition(
        name="research",
        description="Search knowledge base and web for information",
        tool_sequence=["knowledge_search", "web_search"],
        persona="researcher",
    ),
    SkillDefinition(
        name="analyze_data",
        description="Query data and compute statistics",
        tool_sequence=["knowledge_search", "compute_stats"],
        persona="analyst",
    ),
    SkillDefinition(
        name="write_and_send",
        description="Draft content and send via email",
        tool_sequence=["knowledge_search", "file_write", "send_email"],
        persona="communicator",
    ),
]
