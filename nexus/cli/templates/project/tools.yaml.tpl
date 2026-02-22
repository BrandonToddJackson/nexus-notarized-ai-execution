# tools.yaml — declarative tool configuration for {{project_name}}
# Override built-in tool settings or declare custom tool metadata.
# Implementations are registered via the @tool decorator in your code.

tools:
  # Built-in tools (override risk level, timeout, approval as needed)
  - name: knowledge_search
    description: Search the knowledge base
    risk_level: low
    resource_pattern: "kb:*"
    timeout_seconds: 10
    requires_approval: false

  - name: web_search
    description: Search the web
    risk_level: low
    resource_pattern: "web:*"
    timeout_seconds: 15
    requires_approval: false

  # Custom tool example:
  # - name: my_api_tool
  #   description: Call my internal API
  #   risk_level: medium
  #   resource_pattern: "api:internal:*"
  #   timeout_seconds: 20
  #   requires_approval: false
