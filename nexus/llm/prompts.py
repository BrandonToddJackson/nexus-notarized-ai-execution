"""Every prompt template used by NEXUS. No magic strings anywhere else.

All prompts use .format() with named placeholders.
"""

DECOMPOSE_TASK = """You are a task planner for an AI agent system.
Break the following task into 1-5 concrete steps.
Each step should be a single tool call.

Available tools: {tool_list}
Available personas: {persona_list}

Task: {task}

Respond with a JSON array:
[
  {{"action": "description of step", "tool": "tool_name", "params": {{}}, "persona": "persona_name"}},
  ...
]

Rules:
- Each step uses exactly ONE tool
- Choose the most appropriate persona for each step
- Keep it simple — fewer steps is better
- Parameters should be specific and actionable
"""

DECLARE_INTENT = """You are about to execute a tool call. Declare your intent.

Task context: {task_context}
Retrieved knowledge: {retrieved_context}
Current step: {current_step}
Tool: {tool_name}
Available parameters: {tool_schema}

Respond with JSON:
{{
  "planned_action": "what you intend to do",
  "tool_params": {{}},
  "resource_targets": ["list of resources you\'ll access"],
  "reasoning": "why you chose this action",
  "confidence": 0.0-1.0
}}
"""

SELECT_TOOL = """Given this task step, choose the best tool and parameters.

Step: {step_description}
Available tools: {tool_list}
Context: {context}

Respond with JSON:
{{
  "tool": "tool_name",
  "params": {{}},
  "reasoning": "why this tool"
}}
"""
