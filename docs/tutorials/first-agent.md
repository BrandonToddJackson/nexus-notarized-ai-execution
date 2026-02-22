# Tutorial: Your First NEXUS Agent

Build a custom agent that searches a knowledge base and writes a report — all notarized and auditable.

**Time:** ~15 minutes
**Prerequisites:** Python 3.12, `pip install nexus-ai`

---

## Step 1: Initialize a project

```bash
nexus init my-agent
cd my-agent
```

This creates:
```
my-agent/
  .env            ← API keys and config
  personas.yaml   ← behavioral contracts
  tools.yaml      ← tool configuration
  main.py         ← entry point
  knowledge/      ← drop documents here
```

---

## Step 2: Configure your personas

Edit `personas.yaml`:

```yaml
personas:
  - name: researcher
    description: Searches knowledge base for information
    allowed_tools:
      - knowledge_search
      - file_read
    resource_scopes:
      - "kb:*"
      - "file:read:*"
    intent_patterns:
      - search for information
      - find data about
      - look up
    risk_tolerance: low
    max_ttl_seconds: 60

  - name: writer
    description: Writes reports and summaries
    allowed_tools:
      - file_write
      - knowledge_search
    resource_scopes:
      - "file:write:reports:*"
      - "kb:*"
    intent_patterns:
      - write report
      - create document
      - summarize
    risk_tolerance: low
    max_ttl_seconds: 90
```

---

## Step 3: Add documents to the knowledge base

```bash
# Upload via CLI (requires nexus seed + docker)
nexus knowledge add knowledge/my-doc.txt --namespace research

# Or via API:
curl -X POST http://localhost:8000/v1/knowledge \
  -H "Authorization: nxs_demo_key_12345" \
  -H "Content-Type: application/json" \
  -d '{"content": "...", "namespace": "research", "source": "my-doc.txt"}'
```

---

## Step 4: Run your agent

```bash
nexus run "Research AI safety approaches and write a summary report"
```

Output:
```
Chain: abc123...  Status: COMPLETED  Steps: 2

Seal 1 — knowledge_search [EXECUTED]
  [✓] scope   researcher allowed, kb:research matches kb:*
  [✓] intent  score=0.87 ≥ 0.75  "search for information"
  [✓] ttl     12s active, limit 60s
  [—] drift   SKIP (cold start, no baseline yet)
  Result: Found 3 documents about AI safety frameworks...

Seal 2 — file_write [EXECUTED]
  [✓] scope   writer allowed, file:write:reports:summary.md matches file:write:*
  [✓] intent  score=0.91 ≥ 0.75  "write report"
  [✓] ttl     8s active, limit 90s
  [—] drift   SKIP (cold start)
  Result: Written 1,240 chars to reports/summary.md

Merkle chain: VALID
```

---

## Step 5: Inspect the audit trail

```bash
nexus inspect <chain_id>   # show all seals
nexus verify <chain_id>    # verify Merkle chain integrity
nexus audit --persona researcher --limit 20
```

Or via the dashboard: http://localhost:5173 → Ledger

---

## Step 6: Handle blocked actions

Try adding `send_email` to the researcher persona's task — it will be blocked:

```yaml
# personas.yaml — researcher with wrong tool attempt
personas:
  - name: researcher
    allowed_tools: [knowledge_search]  # no send_email
    resource_scopes: ["kb:*"]
    # ...
```

```bash
nexus run "Search for customer emails and send them a newsletter"
```

```
Seal 1 — send_email [BLOCKED]
  [✗] scope   FAIL: send_email not in allowed_tools [knowledge_search]
  Gate 1 failed — action blocked before execution
```

The blocked attempt is in the ledger. The email was never sent.

---

## Next steps

- [Architecture](../architecture.md) — understand the 4-gate pipeline
- [API Reference](../api-reference.md) — build your own frontend
- [Custom tools](../../examples/custom_tool/main.py) — register domain-specific tools
- [Local LLM](../../examples/local_llm/main.py) — run without cloud API keys
