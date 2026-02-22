# NEXUS Python SDK

Async HTTP client for the NEXUS API.

## Installation

```bash
pip install httpx
# Then copy nexus_client.py to your project, or install nexus[sdk] when published to PyPI
```

## Quick start

```python
import asyncio
from nexus_client import NexusClient

async def main():
    async with NexusClient(
        base_url="http://localhost:8000",
        api_key="nxs_demo_key_12345",
    ) as client:
        # Execute a task (synchronous result)
        result = await client.execute("What is NEXUS?")
        print(result["status"])          # "completed"
        print(len(result["seals"]))      # number of notarized actions

        # Stream a task (real-time SSE events)
        async for event in client.stream("Research AI safety trends"):
            print(event["event"], event.get("data", {}))

asyncio.run(main())
```

## API reference

| Method | Description |
|--------|-------------|
| `execute(task, persona?, tenant_id?)` | Execute task, return completed chain |
| `stream(task, persona?, tenant_id?)` | Async-iterate SSE events |
| `ledger(tenant_id?, chain_id?, persona_id?, limit?, offset?)` | Query ledger |
| `list_personas(tenant_id?)` | List available personas |
| `upload_document(content, namespace?, source?, ...)` | Add to knowledge base |
| `health()` | Check API health |
