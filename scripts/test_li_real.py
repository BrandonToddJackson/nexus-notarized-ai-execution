"""
Quick real-account test for LinkedIn connection request.
Run from project root:
  LI_AT=$(grep '^LI_AT=' .env | cut -d= -f2-) LI_CSRF=$(grep '^LI_CSRF=' .env | cut -d= -f2-) \\
    .venv312/bin/python scripts/test_li_real.py
"""
import asyncio
import json
import os
import sys

if not os.getenv("LI_AT"):
    print("ERROR: LI_AT not set.")
    sys.exit(1)

# Unset env vars that break pydantic when nexus.config loads
# (CORS_ORIGINS is a JSON list — pydantic chokes if it's malformed)
for _k in ["CORS_ORIGINS", "ALLOWED_HOSTS", "TRUSTED_HOSTS"]:
    os.environ.pop(_k, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nexus.tools.builtin.sales_growth import (
    linkedin_get_profile,
    linkedin_send_connection_request,
    craft_linkedin_message,
)

LINKEDIN_URL = "https://www.linkedin.com/in/ganitagya"


async def main():
    print(f"Target: {LINKEDIN_URL}\n")

    # ── Step 1: profile lookup ───────────────────────────────────────────────
    print("Step 1 — Profile lookup (dash endpoint)...")
    profile = await linkedin_get_profile(LINKEDIN_URL)

    if profile.get("_stub"):
        print("ERROR: Got stub — cookies not working.")
        sys.exit(1)

    print(json.dumps(profile, indent=2))

    first_name = profile.get("first_name") or "there"
    headline   = profile.get("headline", "")
    summary    = profile.get("summary", "")

    # ── Step 2: craft note via LLM (Ollama / Anthropic) ─────────────────────
    print("\nStep 2 — Crafting personalized note via LLM...")
    note = await craft_linkedin_message(
        first_name=first_name,
        engagement_status="Interested",
        headline=headline,
        summary=summary,
    )
    print(f"Note ({len(note)} chars):\n  {note!r}")

    # ── Step 3: confirm + send ───────────────────────────────────────────────
    print(f"\nReady to send connection request.")
    confirm = input("Send? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    print("\nStep 3 — Sending...")
    result = await linkedin_send_connection_request(LINKEDIN_URL, note=note)
    print(json.dumps(result, indent=2))

    status = result.get("status")
    if status == "sent":
        print(f"\nSent! Log: ~/.nexus/li_sent_log.jsonl")
    elif status == "skipped":
        print(f"\nSkipped: {result.get('reason')}")
    else:
        print(f"\nFailed: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())
