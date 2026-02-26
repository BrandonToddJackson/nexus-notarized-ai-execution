"""shadcn/ui frontend design generator demo.

Full pipeline:
  1. Connect to shadcn MCP server (npx shadcn@latest mcp)
  2. Search for UI components matching the user's prompt
  3. Get example code and add command from the registry
  4. Generate TSX scaffold + static HTML preview via Phase 21 JS sandbox
  5. Save output files to examples/shadcn_frontend/output/

Usage:
    python examples/shadcn_frontend/main.py
    python examples/shadcn_frontend/main.py "SaaS invoice management" --color violet
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# ── Bootstrap path ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NEXUS_GATE_INTENT_THRESHOLD", "0.3")
os.environ.setdefault("NEXUS_SECRET_KEY", "demo-secret-key-for-shadcn-example-32b")

OUTPUT_DIR = Path(__file__).parent / "output"


async def shadcn_frontend_pipeline(
    user_prompt: str,
    color_scheme: str = "zinc",
) -> None:
    """Full shadcn MCP + frontend design generation pipeline."""
    from nexus.mcp.client import MCPClient
    from nexus.types import MCPServerConfig
    from nexus.tools.builtin.frontend_design import generate_frontend_design, _detect_app_type

    print(f"\n{'='*60}")
    print("  shadcn/ui Frontend Design Generator")
    print(f"{'='*60}")
    print(f"  Prompt      : {user_prompt}")
    print(f"  Color scheme: {color_scheme}")
    print(f"{'='*60}\n")

    # ── Step 1: Connect to shadcn MCP ─────────────────────────────────────────
    print("Step 1: Connecting to shadcn MCP server...")
    client = MCPClient()
    cfg = MCPServerConfig(
        id="shadcn",
        tenant_id="demo",
        name="shadcn",
        url="",
        transport="stdio",
        command="npx",
        args=["shadcn@latest", "mcp"],
    )
    await client.connect(cfg)
    session = list(client._sessions.values())[0]
    print("  ✓ Connected to npx shadcn@latest mcp")

    try:
        # ── Step 2: Detect app type + search for components ───────────────────
        app_type = _detect_app_type(user_prompt)
        print(f"\nStep 2: App type detected → {app_type!r}")

        # Map app_type to search terms
        search_terms = {
            "dashboard": "chart sidebar card",
            "crud":      "table data dialog",
            "auth":      "login form card",
            "landing":   "hero navigation card",
            "form":      "form input wizard",
            "settings":  "settings form switch",
            "kanban":    "kanban board drag",
            "inbox":     "email inbox sidebar",
        }.get(app_type, "dashboard card chart")

        print(f"         Searching registry: {search_terms!r}")
        search_result = await session.call_tool(
            "search_items_in_registries",
            {"registries": ["@shadcn"], "query": search_terms, "limit": 6},
        )
        search_text = search_result.content[0].text if search_result.content else ""
        print(f"  ✓ Found components:\n    {search_text[:300].strip()}")

        # ── Step 3: Get add command for core components ───────────────────────
        from nexus.tools.builtin.frontend_design import _APP_COMPONENTS
        core_components = _APP_COMPONENTS.get(app_type, _APP_COMPONENTS["dashboard"])[:6]

        print(f"\nStep 3: Getting install command for: {core_components}")
        add_result = await session.call_tool(
            "get_add_command_for_items",
            {"items": core_components},
        )
        add_command = add_result.content[0].text.strip() if add_result.content else f"npx shadcn@latest add {' '.join(core_components)}"
        print(f"  ✓ Install command: {add_command}")

        # ── Step 4: Get component examples ───────────────────────────────────
        example_component = "card-demo" if app_type in ("dashboard", "landing") else "data-table-demo"
        print(f"\nStep 4: Fetching example code for {example_component!r}...")
        ex_result = await session.call_tool(
            "get_item_examples_from_registries",
            {"registries": ["@shadcn"], "query": example_component},
        )
        examples_text = ex_result.content[0].text if ex_result.content else ""
        print(f"  ✓ Got {len(examples_text)} chars of example TSX")

        # ── Step 5: Get audit checklist ───────────────────────────────────────
        print("\nStep 5: Fetching shadcn audit checklist...")
        audit_result = await session.call_tool("get_audit_checklist", {})
        audit_text = audit_result.content[0].text if audit_result.content else ""
        print(f"  ✓ Audit checklist retrieved ({len(audit_text)} chars)")

    finally:
        await client.disconnect_all()
        print("  ✓ Disconnected from shadcn MCP\n")

    # ── Step 6: Generate frontend design ──────────────────────────────────────
    print("Step 6: Generating TSX + HTML preview via NEXUS JS sandbox...")
    result = await generate_frontend_design(
        prompt=user_prompt,
        components=core_components,
        app_type=app_type,
        color_scheme=color_scheme,
        add_command=add_command,
        component_examples=examples_text[:2000] if examples_text else None,
    )
    print(f"  ✓ TSX code:    {len(result['tsx_code'])} chars")
    print(f"  ✓ HTML preview:{len(result['html_preview'])} chars")
    print(f"  ✓ Components:  {result['components_used']}")
    print(f"  ✓ Install:     {result['install_command']}")

    # ── Step 7: Save output files ─────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = "_".join(user_prompt.lower().split()[:4]).replace("/", "_")

    tsx_path  = OUTPUT_DIR / f"{safe}.tsx"
    html_path = OUTPUT_DIR / f"{safe}.html"
    meta_path = OUTPUT_DIR / f"{safe}_meta.json"

    tsx_path.write_text(result["tsx_code"])
    html_path.write_text(result["html_preview"])
    meta_path.write_text(json.dumps({
        "prompt":        user_prompt,
        "app_type":      result["app_type"],
        "color_scheme":  color_scheme,
        "components":    result["components_used"],
        "install":       result["install_command"],
        "audit_checklist_length": len(audit_text),
        "files": {
            "tsx":  str(tsx_path),
            "html": str(html_path),
        },
    }, indent=2))

    print(f"\n{'='*60}")
    print("  Output files:")
    print(f"  TSX   → {tsx_path}")
    print(f"  HTML  → {html_path}")
    print(f"  Meta  → {meta_path}")
    print(f"{'='*60}")
    print(f"\n  To preview: open {html_path}")
    print(f"  To install: {add_command}\n")

    # ── Step 8: Print TSX preview ──────────────────────────────────────────────
    print("─── Generated TSX (first 800 chars) ───")
    print(result["tsx_code"][:800])
    print("...")

    # ── Audit checklist snippet ────────────────────────────────────────────────
    print("\n─── shadcn Audit Checklist (first 400 chars) ───")
    print(audit_text[:400])


async def multi_design_showcase() -> None:
    """Generate 3 different product designs to show breadth."""
    demos = [
        ("SaaS analytics dashboard for a B2B startup", "zinc"),
        ("Invoice management system for freelancers",  "blue"),
        ("User authentication and login screen",       "violet"),
    ]
    for prompt, color in demos:
        await shadcn_frontend_pipeline(prompt, color)
        print("\n" + "─" * 60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        prompt = " ".join(sys.argv[1:])
        color  = "zinc"
        if "--color" in sys.argv:
            ci = sys.argv.index("--color")
            if ci + 1 < len(sys.argv):
                color = sys.argv[ci + 1]
                prompt = " ".join(a for a in sys.argv[1:] if a not in ("--color", color))
        asyncio.run(shadcn_frontend_pipeline(prompt.strip(), color))
    else:
        asyncio.run(multi_design_showcase())
