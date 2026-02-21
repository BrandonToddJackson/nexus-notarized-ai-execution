"""NEXUS Quickstart — Minimal 10-line usage.

Run: python examples/quickstart/main.py

This demonstrates the core NEXUS loop:
1. Create engine
2. Run a task
3. See the notarized seal
"""

import asyncio

async def main():
    # TODO: Import and configure NEXUS
    # from nexus.core.engine import NexusEngine
    # ... setup components ...
    # result = await engine.run("What is NEXUS?", "demo")
    # print(result)
    print("NEXUS Quickstart")
    print("================")
    print("TODO: Full implementation — run after Phase 4 is complete")
    print()
    print("Expected output:")
    print("  Chain ID: abc-123")
    print("  Steps: 1")
    print("  Seal: {id: ..., tool: knowledge_search, status: executed, gates: [pass, pass, pass, skip]}")


if __name__ == "__main__":
    asyncio.run(main())
