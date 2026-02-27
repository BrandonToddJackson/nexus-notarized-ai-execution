"""Instantly.ai API tools — pre-wired so the LLM never guesses params.

Auth is injected from config.instantly_api_key. All tools use **kwargs
so unexpected LLM-generated params never cause a TypeError.

Tools:
  instantly_get_campaigns          — list all campaigns + status
  instantly_get_campaign_analytics — send/open/reply stats for one campaign
  instantly_get_leads              — leads in a campaign
  instantly_audit                  — full health report: capacity, dupes, sender health
  instantly_add_leads              — add leads to a campaign (with dedup)
  instantly_get_sender_health      — warmup status of all sending accounts
  instantly_move_leads             — move leads from one list to another
"""

import httpx
from nexus.exceptions import ToolError
from nexus.types import ToolDefinition, RiskLevel
from nexus.tools.plugin import _registered_tools

_BASE = "https://api.instantly.ai/api/v2"


def _headers() -> dict:
    from nexus.config import config
    key = config.instantly_api_key
    if not key:
        raise ToolError("instantly_api_key not configured. Set INSTANTLY_API_KEY in your .env file.")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _check_auth(r) -> None:
    if r.status_code == 401:
        raise ToolError("Instantly API key is invalid or expired. Update INSTANTLY_API_KEY in your .env file.")


# ── Read tools ────────────────────────────────────────────────────────────────

async def instantly_get_campaigns(**kwargs) -> dict:
    """Fetch all Instantly campaigns with their status and basic stats."""
    limit = int(kwargs.get("limit") or 10)
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{_BASE}/campaigns", params={"limit": limit}, headers=_headers())
    _check_auth(r)
    r.raise_for_status()
    return r.json()


async def instantly_get_campaign_analytics(**kwargs) -> dict:
    """Get send/open/reply analytics for a specific Instantly campaign."""
    campaign_id = kwargs.get("campaign_id") or ""
    if not campaign_id:
        raise ToolError("campaign_id is required")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(
            f"{_BASE}/campaigns/analytics",
            params={"id": campaign_id, "exclude_total_leads_count": True},
            headers=_headers(),
        )
    _check_auth(r)
    r.raise_for_status()
    data = r.json()
    # Normalize: API returns list or dict with items
    if isinstance(data, list):
        return {"analytics": data[0] if len(data) == 1 else data}
    items = data.get("items") or []
    return {"analytics": items[0] if len(items) == 1 else data}


async def instantly_get_leads(**kwargs) -> dict:
    """Get leads for a specific Instantly campaign."""
    campaign_id = kwargs.get("campaign_id") or ""
    limit = int(kwargs.get("limit") or 50)
    if not campaign_id:
        raise ToolError("campaign_id is required")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{_BASE}/leads/list",
            json={"campaign": campaign_id, "limit": limit},
            headers=_headers(),
        )
    _check_auth(r)
    r.raise_for_status()
    return r.json()


async def instantly_get_sender_health(**kwargs) -> dict:
    """Get warmup status and health of all Instantly sending accounts."""
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{_BASE}/accounts", params={"limit": 100, "status": 1}, headers=_headers())
    _check_auth(r)
    r.raise_for_status()
    accounts = r.json().get("items") or []

    warmed = [a for a in accounts if a.get("warmup_status") == 1]
    unwarmed = [a for a in accounts if a.get("warmup_status") != 1]

    # Theoretical daily capacity: 30 emails/sender, 9.5h window, ~14min avg gap
    max_per_sender = min(30, 570 // 14)
    daily_capacity = max_per_sender * len(warmed)

    return {
        "total_accounts": len(accounts),
        "warmed": len(warmed),
        "unwarmed": len(unwarmed),
        "daily_capacity": daily_capacity,
        "warmed_senders": [
            {
                "email": a.get("email"),
                "tracking_domain": a.get("tracking_domain_name"),
            }
            for a in warmed
        ],
        "unwarmed_senders": [a.get("email") for a in unwarmed],
    }


# ── Audit tool ────────────────────────────────────────────────────────────────

async def instantly_audit(**kwargs) -> dict:
    """Full campaign health report: capacity, duplicate leads, sender health.

    Covers:
    1. All campaigns with lead counts and status
    2. Cross-campaign duplicate lead detection
    3. Sender warmup health + daily capacity
    4. Recommendations
    """
    from collections import Counter

    headers = _headers()

    async with httpx.AsyncClient(timeout=60) as c:
        # 1. Campaigns
        r = await c.get(f"{_BASE}/campaigns", params={"limit": 100}, headers=headers)
        _check_auth(r)
        r.raise_for_status()
        campaigns = r.json().get("items") or []

        # 2. Leads per campaign (for duplicate check)
        campaign_leads: dict[str, list[str]] = {}
        for camp in campaigns:
            cid = camp.get("id", "")
            leads_r = await c.post(
                f"{_BASE}/leads/list",
                json={"campaign": cid, "limit": 500},
                headers=headers,
            )
            if leads_r.status_code == 200:
                items = leads_r.json().get("items") or []
                campaign_leads[cid] = [
                    (item.get("email") or "").strip().lower()
                    for item in items
                    if item.get("email")
                ]
            else:
                campaign_leads[cid] = []

        # 3. Sender accounts
        acc_r = await c.get(f"{_BASE}/accounts", params={"limit": 100, "status": 1}, headers=headers)
        accounts = acc_r.json().get("items") or [] if acc_r.status_code == 200 else []

    # Duplicate analysis
    all_emails = [e for emails in campaign_leads.values() for e in emails]
    counts = Counter(all_emails)
    duplicates = {email: cnt for email, cnt in counts.items() if cnt > 1}

    # Sender health
    warmed = [a for a in accounts if a.get("warmup_status") == 1]
    n_warmed = len(warmed)
    max_per_sender = min(30, 570 // 14)
    daily_capacity = max_per_sender * n_warmed

    # Build campaign summary
    status_map = {0: "draft", 1: "active", 2: "paused", 3: "completed"}
    campaign_summary = [
        {
            "id": c.get("id"),
            "name": c.get("name"),
            "status": status_map.get(c.get("status"), "unknown"),
            "lead_count": len(campaign_leads.get(c.get("id", ""), [])),
        }
        for c in campaigns
    ]

    total_leads = sum(len(v) for v in campaign_leads.values())

    # Recommendations
    recommendations = []
    if duplicates:
        recommendations.append(f"{len(duplicates)} leads appear in multiple campaigns — run dedup")
    if total_leads < 50:
        recommendations.append("Lead pool is small — add more leads via SuperSearch")
    if n_warmed < 3:
        recommendations.append(f"Only {n_warmed} warmed sender(s) — warm more accounts for throughput")
    if not recommendations:
        recommendations.append("Setup looks healthy")

    return {
        "campaigns": campaign_summary,
        "total_campaigns": len(campaigns),
        "total_leads": total_leads,
        "duplicate_leads": len(duplicates),
        "top_duplicates": [
            {"email": e, "count": cnt}
            for e, cnt in sorted(duplicates.items(), key=lambda x: -x[1])[:10]
        ],
        "senders": {
            "total": len(accounts),
            "warmed": n_warmed,
            "daily_capacity": daily_capacity,
        },
        "recommendations": recommendations,
    }


# ── Write tools ───────────────────────────────────────────────────────────────

async def instantly_create_campaign(**kwargs) -> dict:
    """Create a new Instantly campaign with email sequence and warmed senders.

    The LLM provides the campaign name and email copy; this tool handles
    fetching warmed senders, building the schedule/settings payload, and
    calling POST /campaigns. Idempotent: returns existing campaign if name matches.

    Required: name, subject, body
    Optional: follow_up_subject, follow_up_body, follow_up_delay_days (default 3),
              daily_limit (default 30), timezone (default America/Chicago)
    """
    name = kwargs.get("name") or ""
    subject = kwargs.get("subject") or ""
    body = kwargs.get("body") or ""
    if not name:
        raise ToolError("name is required")
    if not subject:
        raise ToolError("subject is required")
    if not body:
        raise ToolError("body is required")

    follow_up_subject = kwargs.get("follow_up_subject") or ""
    follow_up_body = kwargs.get("follow_up_body") or ""
    follow_up_delay = int(kwargs.get("follow_up_delay_days") or 3)
    daily_limit = int(kwargs.get("daily_limit") or 30)
    timezone = kwargs.get("timezone") or "America/Chicago"

    from datetime import date

    headers = _headers()

    async with httpx.AsyncClient(timeout=60) as c:
        # Check if campaign already exists by name
        search_r = await c.get(
            f"{_BASE}/campaigns",
            params={"limit": 100, "search": name[:50]},
            headers=headers,
        )
        _check_auth(search_r)
        search_r.raise_for_status()
        for item in (search_r.json().get("items") or []):
            if (item.get("name") or "").strip() == name:
                return {
                    "campaign_id": item["id"],
                    "status": "existing",
                    "name": name,
                    "message": "Campaign already exists — returned existing ID",
                }

        # Fetch warmed senders
        acc_r = await c.get(
            f"{_BASE}/accounts",
            params={"limit": 100, "status": 1},
            headers=headers,
        )
        _check_auth(acc_r)
        acc_r.raise_for_status()
        accounts = acc_r.json().get("items") or []
        warmed_emails = [
            a["email"] for a in accounts
            if a.get("warmup_status") == 1 and a.get("email")
        ]
        if not warmed_emails:
            raise ToolError(
                "No warmed sender accounts found. "
                "Add and warm senders in Instantly before creating a campaign."
            )

        # Build sequence steps
        steps = [{"type": "email", "delay": 0, "variants": [{"subject": subject, "body": body}]}]
        if follow_up_subject and follow_up_body:
            steps.append({
                "type": "email",
                "delay": follow_up_delay,
                "variants": [{"subject": follow_up_subject, "body": follow_up_body}],
            })

        today = date.today().isoformat()
        payload = {
            "name": name,
            "email_list": warmed_emails,
            "campaign_schedule": {
                "start_date": today,
                "end_date": None,
                "schedules": [{
                    "name": "Weekdays",
                    "timing": {"from": "08:30", "to": "18:00"},
                    "days": {"0": False, "1": True, "2": True, "3": True, "4": True, "5": True, "6": False},
                    "timezone": timezone,
                }],
            },
            "sequences": [{"steps": steps}],
            "email_gap": 10,
            "random_wait_max": 8,
            "daily_limit": daily_limit,
            "daily_max_leads": 50,
            "stop_on_reply": True,
            "stop_on_auto_reply": True,
            "link_tracking": True,
            "open_tracking": True,
            "insert_unsubscribe_header": True,
            "match_lead_esp": True,
            "stop_for_company": True,
            "prioritize_new_leads": True,
        }

        r = await c.post(f"{_BASE}/campaigns", json=payload, headers=headers)
        _check_auth(r)
        r.raise_for_status()
        data = r.json()

    campaign_id = data.get("id") or ""
    return {
        "campaign_id": campaign_id,
        "status": "created",
        "name": name,
        "senders": len(warmed_emails),
        "sequence_steps": len(steps),
        "message": f"Campaign '{name}' created with {len(warmed_emails)} warmed senders. "
                   "Add leads then call instantly_activate_campaign to launch.",
    }


async def instantly_activate_campaign(**kwargs) -> dict:
    """Activate (launch) an Instantly campaign by ID.

    Changes campaign status from draft to active — emails will start sending.
    Call this after adding leads to the campaign.
    """
    campaign_id = kwargs.get("campaign_id") or ""
    if not campaign_id:
        raise ToolError("campaign_id is required")

    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{_BASE}/campaigns/{campaign_id}/activate",
            json={},
            headers=_headers(),
        )
    _check_auth(r)
    r.raise_for_status()

    return {
        "campaign_id": campaign_id,
        "status": "activated",
        "message": "Campaign is now active. Emails will begin sending on the next scheduled window.",
    }


async def instantly_add_leads(**kwargs) -> dict:
    """Add one or more leads to an Instantly campaign.

    Accepts either a single lead (email + optional fields) or a list of leads.
    Skips leads already in the campaign (dedup).
    """
    campaign_id = kwargs.get("campaign_id") or ""
    if not campaign_id:
        raise ToolError("campaign_id is required")

    # Accept single lead or list
    leads_input = kwargs.get("leads") or []
    if not leads_input:
        # Single lead shorthand
        email = kwargs.get("email") or ""
        if not email:
            raise ToolError("Either 'leads' list or 'email' is required")
        leads_input = [{
            "email": email,
            "first_name": kwargs.get("first_name") or "",
            "last_name": kwargs.get("last_name") or "",
            "company_name": kwargs.get("company_name") or "",
        }]

    # Fetch existing leads for dedup
    async with httpx.AsyncClient(timeout=60) as c:
        existing_r = await c.post(
            f"{_BASE}/leads/list",
            json={"campaign": campaign_id, "limit": 1000},
            headers=_headers(),
        )
        existing_emails: set[str] = set()
        if existing_r.status_code == 200:
            items = existing_r.json().get("items") or []
            existing_emails = {(i.get("email") or "").lower() for i in items}

        added = 0
        skipped = 0
        errors = []
        for lead in leads_input:
            email = (lead.get("email") or "").strip().lower()
            if not email:
                continue
            if email in existing_emails:
                skipped += 1
                continue
            payload = {
                "campaign_id": campaign_id,
                "email": email,
                "first_name": lead.get("first_name") or "",
                "last_name": lead.get("last_name") or "",
                "company_name": lead.get("company_name") or "",
            }
            r = await c.post(f"{_BASE}/leads", json=payload, headers=_headers())
            _check_auth(r)
            if r.status_code in (200, 201):
                added += 1
                existing_emails.add(email)
            elif r.status_code in (400, 409, 422):
                skipped += 1  # already exists or invalid
            else:
                errors.append({"email": email, "status": r.status_code})

    return {
        "campaign_id": campaign_id,
        "added": added,
        "skipped_duplicates": skipped,
        "errors": errors,
    }


async def instantly_move_leads(**kwargs) -> dict:
    """Move leads from one Instantly list to another list or campaign."""
    from_list_id = kwargs.get("from_list_id") or ""
    to_list_id = kwargs.get("to_list_id") or ""
    to_campaign_id = kwargs.get("to_campaign_id") or ""

    if not from_list_id:
        raise ToolError("from_list_id is required")
    if not to_list_id and not to_campaign_id:
        raise ToolError("Either to_list_id or to_campaign_id is required")

    payload: dict = {"list_id": from_list_id, "copy_leads": bool(kwargs.get("copy", False))}
    if to_list_id:
        payload["to_list_id"] = to_list_id
    if to_campaign_id:
        payload["to_campaign_id"] = to_campaign_id

    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{_BASE}/leads/move", json=payload, headers=_headers())
    _check_auth(r)
    r.raise_for_status()
    data = r.json()

    moved = data.get("leads_moved") or data.get("moved") or data.get("total_count") or 0
    is_async = data.get("type") == "move-leads" or data.get("status") == "pending"

    return {
        "from_list_id": from_list_id,
        "moved": moved if not is_async else "async_job_started",
        "job_id": data.get("id") if is_async else None,
    }


# ── Tool registrations ────────────────────────────────────────────────────────

_registered_tools["instantly_get_campaigns"] = (
    ToolDefinition(
        name="instantly_get_campaigns",
        description="Fetch all Instantly.ai campaigns with status and lead counts",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max campaigns to return (default 10)"},
            },
        },
        risk_level=RiskLevel.LOW,
        resource_pattern="http:*",
        timeout_seconds=30,
        requires_approval=False,
    ),
    instantly_get_campaigns,
)

_registered_tools["instantly_get_campaign_analytics"] = (
    ToolDefinition(
        name="instantly_get_campaign_analytics",
        description="Get send/open/reply analytics for a specific Instantly campaign",
        parameters={
            "type": "object",
            "properties": {
                "campaign_id": {"type": "string", "description": "Instantly campaign ID"},
            },
            "required": ["campaign_id"],
        },
        risk_level=RiskLevel.LOW,
        resource_pattern="http:*",
        timeout_seconds=30,
        requires_approval=False,
    ),
    instantly_get_campaign_analytics,
)

_registered_tools["instantly_get_leads"] = (
    ToolDefinition(
        name="instantly_get_leads",
        description="Get leads for a specific Instantly campaign",
        parameters={
            "type": "object",
            "properties": {
                "campaign_id": {"type": "string", "description": "Instantly campaign ID"},
                "limit": {"type": "integer", "description": "Max leads to return (default 50)"},
            },
            "required": ["campaign_id"],
        },
        risk_level=RiskLevel.LOW,
        resource_pattern="http:*",
        timeout_seconds=30,
        requires_approval=False,
    ),
    instantly_get_leads,
)

_registered_tools["instantly_audit"] = (
    ToolDefinition(
        name="instantly_audit",
        description=(
            "Full Instantly.ai campaign health report: all campaigns with lead counts, "
            "cross-campaign duplicate detection, sender warmup health, daily capacity, "
            "and actionable recommendations"
        ),
        parameters={
            "type": "object",
            "properties": {},
        },
        risk_level=RiskLevel.LOW,
        resource_pattern="http:*",
        timeout_seconds=60,
        requires_approval=False,
    ),
    instantly_audit,
)

_registered_tools["instantly_get_sender_health"] = (
    ToolDefinition(
        name="instantly_get_sender_health",
        description="Get warmup status and daily sending capacity for all Instantly sender accounts",
        parameters={
            "type": "object",
            "properties": {},
        },
        risk_level=RiskLevel.LOW,
        resource_pattern="http:*",
        timeout_seconds=30,
        requires_approval=False,
    ),
    instantly_get_sender_health,
)

_registered_tools["instantly_add_leads"] = (
    ToolDefinition(
        name="instantly_add_leads",
        description="Add leads to an Instantly campaign with automatic deduplication",
        parameters={
            "type": "object",
            "properties": {
                "campaign_id": {"type": "string", "description": "Target campaign ID"},
                "email": {"type": "string", "description": "Single lead email (shorthand)"},
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "company_name": {"type": "string"},
                "leads": {
                    "type": "array",
                    "description": "List of lead objects with email, first_name, last_name, company_name",
                    "items": {"type": "object"},
                },
            },
            "required": ["campaign_id"],
        },
        risk_level=RiskLevel.MEDIUM,
        resource_pattern="http:*",
        timeout_seconds=60,
        requires_approval=False,
    ),
    instantly_add_leads,
)

_registered_tools["instantly_move_leads"] = (
    ToolDefinition(
        name="instantly_move_leads",
        description="Move leads from one Instantly list to another list or campaign",
        parameters={
            "type": "object",
            "properties": {
                "from_list_id": {"type": "string", "description": "Source list ID"},
                "to_list_id": {"type": "string", "description": "Destination list ID"},
                "to_campaign_id": {"type": "string", "description": "Destination campaign ID"},
                "copy": {"type": "boolean", "description": "Copy instead of move (default false)"},
            },
            "required": ["from_list_id"],
        },
        risk_level=RiskLevel.MEDIUM,
        resource_pattern="http:*",
        timeout_seconds=60,
        requires_approval=False,
    ),
    instantly_move_leads,
)

_registered_tools["instantly_create_campaign"] = (
    ToolDefinition(
        name="instantly_create_campaign",
        description=(
            "Create a new Instantly.ai email campaign with warmed senders and email sequence. "
            "Idempotent — returns existing campaign ID if name already exists. "
            "After creating, add leads with instantly_add_leads then activate with instantly_activate_campaign."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Campaign name"},
                "subject": {"type": "string", "description": "Email subject line for step 1"},
                "body": {"type": "string", "description": "Email body for step 1 (plain text, supports {{first_name}} etc.)"},
                "follow_up_subject": {"type": "string", "description": "Subject line for follow-up email (optional)"},
                "follow_up_body": {"type": "string", "description": "Body for follow-up email (optional)"},
                "follow_up_delay_days": {"type": "integer", "description": "Days to wait before follow-up (default 3)"},
                "daily_limit": {"type": "integer", "description": "Max emails per day (default 30)"},
                "timezone": {"type": "string", "description": "Schedule timezone (default America/Chicago)"},
            },
            "required": ["name", "subject", "body"],
        },
        risk_level=RiskLevel.HIGH,
        resource_pattern="http:*",
        timeout_seconds=60,
        requires_approval=True,
    ),
    instantly_create_campaign,
)

_registered_tools["instantly_activate_campaign"] = (
    ToolDefinition(
        name="instantly_activate_campaign",
        description="Activate (launch) an Instantly campaign. Emails will begin sending on the next scheduled window.",
        parameters={
            "type": "object",
            "properties": {
                "campaign_id": {"type": "string", "description": "Instantly campaign ID to activate"},
            },
            "required": ["campaign_id"],
        },
        risk_level=RiskLevel.HIGH,
        resource_pattern="http:*",
        timeout_seconds=30,
        requires_approval=True,
    ),
    instantly_activate_campaign,
)
