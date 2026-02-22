# personas.yaml — behavioral contracts for {{project_name}}
# Edit to restrict what the agent can do in each mode.
# Full reference: https://github.com/nexus-ai/nexus/docs/personas.md

personas:
  - name: researcher
    description: Searches and retrieves information
    allowed_tools:
      - knowledge_search
      - web_search
      - file_read
    resource_scopes:
      - "kb:*"
      - "web:*"
      - "file:read:*"
    intent_patterns:
      - search for information
      - find data about
      - look up
    risk_tolerance: low
    max_ttl_seconds: 60

  # Add more personas below:
  # - name: analyst
  #   description: Analyzes data
  #   allowed_tools: [compute_stats, file_read]
  #   resource_scopes: ["data:*", "file:read:*"]
  #   intent_patterns: [analyze, calculate]
  #   risk_tolerance: medium
  #   max_ttl_seconds: 120
