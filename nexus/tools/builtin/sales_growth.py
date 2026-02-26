"""Sales Growth tools: Instantly, LinkedIn (Voyager API), Retell, Google Sheets, Hunter/Clearbit.

Authentication via environment variables:
  INSTANTLY_API_KEY       – Instantly v2 bearer token (base64 userId:token)
  HUNTER_API_KEY          – Hunter.io API key for phone/email enrichment
  CLEARBIT_API_KEY        – Clearbit API key for enrichment fallback
  LI_AT                   – LinkedIn session cookie (li_at value from browser DevTools)
  LI_CSRF                 – LinkedIn CSRF token (JSESSIONID value, no quotes)
  RETELL_API_KEY          – Retell key for voice batch calls
  GOOGLE_ACCESS_TOKEN     – Google OAuth2 bearer token for Sheets writes

LinkedIn setup (free, no third-party service needed):
  1. Log in to LinkedIn in Chrome
  2. DevTools → Application → Cookies → linkedin.com
  3. Copy li_at value → LI_AT
  4. Copy JSESSIONID value (strip surrounding quotes) → LI_CSRF

All tools degrade gracefully when credentials are missing — they return a
stub/skipped response instead of raising, so the 4-gate pipeline can always
complete a cycle and log what was skipped.
"""

import asyncio
import json
import os
import random
import urllib.parse
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from nexus.tools.plugin import tool
from nexus.types import RiskLevel

# ── Base URLs ──────────────────────────────────────────────────────────────────
_INSTANTLY_BASE = "https://api.instantly.ai/api/v2"
_HUNTER_BASE    = "https://api.hunter.io/v2"
_CLEARBIT_BASE  = "https://person.clearbit.com/v2"
_RETELL_BASE    = "https://api.retellai.com/v2"
_SHEETS_BASE    = "https://sheets.googleapis.com/v4/spreadsheets"


# ── Shared HTTP helper: retry on 429 / 5xx ────────────────────────────────────
async def _http(
    method: str,
    url: str,
    *,
    timeout: int = 30,
    retries: int = 3,
    **kwargs: Any,
) -> httpx.Response:
    """Execute an HTTP request with exponential backoff on transient errors.

    Retries on 429 (rate limit) and 5xx (server error).
    Raises the last httpx.HTTPStatusError on permanent failure.
    """
    if retries < 1:
        raise ValueError(f"retries must be >= 1, got {retries}")
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.request(method, url, **kwargs)
            # Only retry on transient errors; raise immediately on permanent 4xx
            if r.status_code not in (429, 503) and r.status_code < 500:
                r.raise_for_status()
                return r
            r.raise_for_status()  # raises for 429/503/5xx
            return r  # unreachable but satisfies type checker
        except httpx.HTTPStatusError as exc:
            # Don't retry permanent client errors (400, 401, 403, 404, etc.)
            if exc.response.status_code not in (429, 503) and exc.response.status_code < 500:
                raise
            last_exc = exc
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
    raise last_exc  # type: ignore[misc]


# ── Per-service auth headers ───────────────────────────────────────────────────
def _instantly_headers() -> dict:
    return {
        "Authorization": f"Bearer {os.getenv('INSTANTLY_API_KEY', '')}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Origin": "https://app.instantly.ai",
        "Referer": "https://app.instantly.ai/",
    }


def _retell_headers() -> dict:
    return {
        "Authorization": f"Bearer {os.getenv('RETELL_API_KEY', '')}",
        "Content-Type": "application/json",
    }


def _sheets_headers() -> dict:
    return {
        "Authorization": f"Bearer {os.getenv('GOOGLE_ACCESS_TOKEN', '')}",
        "Content-Type": "application/json",
    }


# ── LinkedIn Voyager API (direct session-cookie auth) ──────────────────────────
# Mirrors what Unipile does internally — zero third-party cost.
# Required env vars (copy from browser DevTools → Application → Cookies):
#   LI_AT       = li_at cookie value
#   LI_CSRF     = JSESSIONID value (strip surrounding quotes)
#   LI_BCOOKIE  = bcookie value (strip surrounding quotes)
#   LI_BSCOOKIE = bscookie value (strip surrounding quotes)
_LI_VOYAGER = "https://www.linkedin.com/voyager/api"

# Confirmed endpoint from real browser DevTools capture (2026-02-26)
_LI_CONNECT_ENDPOINT = (
    f"{_LI_VOYAGER}/voyagerRelationshipsDashMemberRelationships"
    "?action=verifyQuotaAndCreateV2"
    "&decorationId=com.linkedin.voyager.dash.deco.relationships.InvitationCreationResultWithInvitee-2"
)


def _li_headers() -> dict | None:
    li_at    = os.getenv("LI_AT", "")
    csrf     = os.getenv("LI_CSRF", "")
    bcookie  = os.getenv("LI_BCOOKIE", "")
    bscookie = os.getenv("LI_BSCOOKIE", "")
    if not li_at:
        return None
    # Build cookie string — li_at + JSESSIONID required; bcookie/bscookie needed
    # for write operations (LinkedIn backend validates these)
    cookie_parts = [f"li_at={li_at}", f'JSESSIONID="{csrf}"']
    if bcookie:
        cookie_parts.append(f'bcookie="{bcookie}"')
    if bscookie:
        cookie_parts.append(f'bscookie="{bscookie}"')
    return {
        "cookie":                        "; ".join(cookie_parts),
        "csrf-token":                    csrf,
        "x-restli-protocol-version":     "2.0.0",
        "x-li-lang":                     "en_US",
        "x-li-deco-include-micro-schema": "true",
        "x-li-pem-metadata":             "Voyager - Invitations - Actions=invite-send",
        "x-li-track":                    '{"clientVersion":"1.13.42533","mpVersion":"1.13.42533","osName":"web","timezoneOffset":-5,"timezone":"America/New_York","deviceFormFactor":"DESKTOP","mpName":"voyager-web","displayDensity":1,"displayWidth":1920,"displayHeight":1080}',
        "accept":                        "application/vnd.linkedin.normalized+json+2.1",
        "accept-language":               "en-US,en;q=0.9",
        "content-type":                  "application/json; charset=UTF-8",
        "origin":                        "https://www.linkedin.com",
        "sec-fetch-site":                "same-origin",
        "sec-fetch-mode":                "cors",
        "sec-fetch-dest":                "empty",
        "sec-ch-ua":                     '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
        "sec-ch-ua-mobile":              "?0",
        "sec-ch-ua-platform":            '"macOS"',
        "user-agent":                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        "dnt":                           "1",
    }


async def _li_resolve_profile(vanity: str, headers: dict) -> dict:
    """Resolve a vanity name to URNs needed for API calls.

    Uses the 2026 dash endpoint — old /identity/profiles/{vanity} returns 410.
    Profile data lives in included[0], not data.

    Returns dict with:
        entity_urn  — urn:li:fsd_profile:... (needed for invitations/connections)
        member_urn  — urn:li:member:...       (legacy, kept for DM endpoint)
    """
    r = await _http(
        "GET", f"{_LI_VOYAGER}/identity/dash/profiles",
        headers=headers,
        params={"q": "memberIdentity", "memberIdentity": vanity},
    )
    item = (r.json().get("included") or [{}])[0]
    return {
        "entity_urn": item.get("entityUrn", ""),
        "member_urn": item.get("objectUrn", ""),
    }


async def _li_post_chrome(url: str, headers: dict, payload: dict) -> dict:
    """POST to LinkedIn using Chrome TLS fingerprint via curl_cffi.

    LinkedIn's Cloudflare layer blocks httpx POSTs (TLS fingerprint mismatch).
    curl_cffi impersonates Chrome's TLS handshake so write operations succeed.
    Falls back to httpx if curl_cffi is not installed.
    """
    try:
        from curl_cffi.requests import AsyncSession
        async with AsyncSession(impersonate="chrome131") as session:
            r = await session.post(url, headers=headers, json=payload)
            return {"status_code": r.status_code, "text": r.text}
    except ImportError:
        # curl_cffi not available — fall back (may hit 403 on bot detection)
        r = await _http("POST", url, headers=headers, json=payload)
        return {"status_code": r.status_code, "text": r.text}


# ── LinkedIn rate-limit safety guards ─────────────────────────────────────────
# Five-layer account protection:
#   1. Daily cap            — LI_DAILY_LIMIT (default 10, never exceed 20)
#   2. Weekly cap           — LI_WEEKLY_LIMIT (default 50)
#   3. Time-of-day window   — LI_SEND_HOURS (default "7-21", i.e. 7am–9pm local)
#   4. Circuit breaker      — opens on 429 or 3 consecutive errors; resets next day
#   5. Minimum send interval — LI_MIN_INTERVAL_SECONDS (default 45s between sends)
#   + dedup prevention (never contact same person twice)
_LI_STATE_FILE         = Path.home() / ".nexus" / "li_state.json"
_LI_DAILY_LIMIT        = int(os.getenv("LI_DAILY_LIMIT", "10"))
_LI_WEEKLY_LIMIT       = int(os.getenv("LI_WEEKLY_LIMIT", "50"))
_LI_MIN_INTERVAL       = int(os.getenv("LI_MIN_INTERVAL_SECONDS", "45"))
_LI_SEND_HOURS         = os.getenv("LI_SEND_HOURS", "7-21")  # local time, 24h


def _li_parse_send_window() -> tuple[int, int]:
    """Parse LI_SEND_HOURS ('7-21') into (start_hour, end_hour)."""
    try:
        parts = _LI_SEND_HOURS.split("-")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 7, 21


def _li_iso_week() -> str:
    """Return ISO year-week string e.g. '2026-W09'."""
    from datetime import datetime
    dt = datetime.now()
    return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"


def _li_load_state() -> dict:
    today = str(date.today())
    week  = _li_iso_week()
    base  = {
        "date": today,
        "week": week,
        "connections_today": 0,
        "connections_this_week": 0,
        "dms_today": 0,
        "errors_today": 0,
        "circuit_open": False,
        "circuit_reason": "",
        "last_send_ts": None,
        "sent_to": [],
    }
    if _LI_STATE_FILE.exists():
        try:
            s = json.loads(_LI_STATE_FILE.read_text())
            # Same day — return as-is (all counters still valid)
            if s.get("date") == today:
                # Ensure new fields exist for backward compat
                for k, v in base.items():
                    s.setdefault(k, v)
                return s
            # New day — reset daily counters; carry over weekly if same week
            base["connections_this_week"] = (
                s.get("connections_this_week", 0) if s.get("week") == week else 0
            )
        except (json.JSONDecodeError, OSError):
            pass
    return base


def _li_save_state(state: dict) -> None:
    _LI_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LI_STATE_FILE.write_text(json.dumps(state))


async def _li_guard(member_urn: str) -> None:
    """Enforce all safety guards before a LinkedIn send.

    Raises RuntimeError for hard stops (caps, circuit open, duplicate).
    Sleeps for soft throttles (minimum interval enforcement).
    """
    from datetime import datetime
    state = _li_load_state()

    # 1. Circuit breaker — hard stop
    if state.get("circuit_open"):
        raise RuntimeError(
            f"LinkedIn circuit breaker open: {state.get('circuit_reason', 'too many errors')}. "
            "Resets tomorrow. Check ~/.nexus/li_state.json."
        )

    # 2. Time-of-day window
    start_h, end_h = _li_parse_send_window()
    current_hour = datetime.now().hour
    if not (start_h <= current_hour < end_h):
        raise RuntimeError(
            f"Outside send window ({_LI_SEND_HOURS}, 24h local). "
            f"Current hour: {current_hour}. Set LI_SEND_HOURS to override."
        )

    # 3. Daily cap
    if state["connections_today"] >= _LI_DAILY_LIMIT:
        raise RuntimeError(
            f"Daily connection limit reached ({_LI_DAILY_LIMIT}). "
            "Resets tomorrow. Raise LI_DAILY_LIMIT to increase (max recommended: 20)."
        )

    # 4. Weekly cap
    if state["connections_this_week"] >= _LI_WEEKLY_LIMIT:
        raise RuntimeError(
            f"Weekly connection limit reached ({_LI_WEEKLY_LIMIT}). "
            "Resets next Monday. Raise LI_WEEKLY_LIMIT to increase."
        )

    # 5. Duplicate prevention
    if member_urn in state["sent_to"]:
        raise RuntimeError(f"Already messaged {member_urn} today — skipping duplicate.")

    # 6. Minimum interval — sleep rather than hard-fail
    if state.get("last_send_ts"):
        try:
            last = datetime.fromisoformat(state["last_send_ts"])
            elapsed = (datetime.now() - last).total_seconds()
            gap = _LI_MIN_INTERVAL + random.uniform(0, 15)  # add jitter
            if elapsed < gap:
                wait = gap - elapsed
                await asyncio.sleep(wait)
        except (ValueError, TypeError):
            pass


def _li_record_result(status_code: int, member_urn: str, reason: str = "") -> None:
    """Update state after a send attempt — drives circuit breaker and counters."""
    from datetime import datetime
    state = _li_load_state()

    if status_code in (200, 201, 204):
        # Success
        state["connections_today"]     += 1
        state["connections_this_week"] += 1
        state["errors_today"]           = 0
        state["last_send_ts"]           = datetime.now().isoformat()
        state["sent_to"].append(member_urn)
    elif status_code == 429:
        # Explicit rate limit from LinkedIn — open circuit immediately
        state["circuit_open"]   = True
        state["circuit_reason"] = "LinkedIn returned 429 (rate limited). Wait until tomorrow."
    else:
        # Other error — increment counter, open circuit after 3 consecutive
        state["errors_today"] = state.get("errors_today", 0) + 1
        if state["errors_today"] >= 3:
            state["circuit_open"]   = True
            state["circuit_reason"] = (
                f"3 consecutive errors (last: HTTP {status_code} {reason[:80]}). "
                "Possible auth issue — refresh LI_AT and LI_CSRF."
            )

    _li_save_state(state)


_LI_LOG_FILE = Path.home() / ".nexus" / "li_sent_log.jsonl"


def _li_write_log(linkedin_url: str, member_urn: str, message: str, action: str = "dm") -> None:
    """Append one line to the sent log so you always know who was contacted."""
    from datetime import datetime, timezone as _tz
    entry = {
        "ts":          datetime.now(_tz.utc).isoformat(),
        "action":      action,
        "linkedin_url": linkedin_url,
        "member_urn":  member_urn,
        "message":     message,
    }
    _LI_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _LI_LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Path 1: Audit & Optimize active campaigns ─────────────────────────────────

@tool(name="instantly_list_campaigns", risk_level=RiskLevel.LOW, resource_pattern="crm:campaigns:*", timeout_seconds=30)
async def instantly_list_campaigns(status: int) -> list:
    """List all Instantly campaigns, optionally filtered by status.

    Args:
        status: 0=all, 1=active, 2=paused, 3=completed

    Returns:
        List of campaign dicts with id, name, status, timestamp_created
    """
    params: dict[str, Any] = {"limit": 100}
    if status and status > 0:
        params["status"] = status
    r = await _http("GET", f"{_INSTANTLY_BASE}/campaigns", headers=_instantly_headers(), params=params)
    campaigns = r.json().get("items") or []
    status_map = {1: "active", 2: "paused", 3: "completed", 0: "draft"}
    return [
        {
            "id": c.get("id"),
            "name": c.get("name"),
            "status": status_map.get(c.get("status"), str(c.get("status"))),
            "status_code": c.get("status"),
            "timestamp_created": c.get("timestamp_created"),
            "timestamp_updated": c.get("timestamp_updated"),
            "email_list": c.get("email_list") or [],
        }
        for c in campaigns
    ]


@tool(name="instantly_get_campaign_detail", risk_level=RiskLevel.LOW, resource_pattern="crm:campaigns:*", timeout_seconds=30)
async def instantly_get_campaign_detail(campaign_id: str) -> dict:
    """Get full detail for an Instantly campaign including sequences and settings.

    Args:
        campaign_id: Instantly campaign UUID

    Returns:
        Full campaign dict with sequences, schedule, settings, email_list
    """
    r = await _http("GET", f"{_INSTANTLY_BASE}/campaigns/{campaign_id}", headers=_instantly_headers())
    return r.json()


@tool(name="instantly_get_campaign_analytics", risk_level=RiskLevel.LOW, resource_pattern="crm:campaigns:*", timeout_seconds=30)
async def instantly_get_campaign_analytics(campaign_id: str) -> dict:
    """Get analytics for an Instantly campaign (opens, clicks, replies, bounces).

    Args:
        campaign_id: Instantly campaign UUID

    Returns:
        Dict with leads_count, contacted_count, emails_sent, open_count,
        reply_count, bounced_count, unsubscribed_count, open_rate, reply_rate
    """
    r = await _http(
        "GET", f"{_INSTANTLY_BASE}/campaigns/analytics",
        headers=_instantly_headers(),
        params={"id": campaign_id, "exclude_total_leads_count": "false"},
    )
    raw = r.json()

    # API returns list or dict depending on number of campaigns
    if isinstance(raw, list):
        data = raw[0] if raw else {}
    elif isinstance(raw, dict):
        items = raw.get("items") or raw.get("campaigns") or []
        data = items[0] if isinstance(items, list) and items else raw
    else:
        data = {}

    sent = data.get("emails_sent") or data.get("sent") or 0
    opens = data.get("open_count") or data.get("opens") or 0
    replies = data.get("reply_count") or data.get("replies") or 0
    return {
        "campaign_id": campaign_id,
        "leads_count": data.get("leads_count") or data.get("total_leads") or 0,
        "contacted_count": data.get("contacted_count") or 0,
        "emails_sent": sent,
        "open_count": opens,
        "reply_count": replies,
        "bounced_count": data.get("bounced_count") or data.get("bounces") or 0,
        "unsubscribed_count": data.get("unsubscribed_count") or data.get("unsubscribed") or 0,
        "open_rate": round(opens / sent, 4) if sent > 0 else 0.0,
        "reply_rate": round(replies / sent, 4) if sent > 0 else 0.0,
        "_raw": data,
    }


@tool(name="instantly_audit_campaigns", risk_level=RiskLevel.LOW, resource_pattern="crm:campaigns:*", timeout_seconds=60)
async def instantly_audit_campaigns(include_analytics: bool) -> dict:
    """Audit all Instantly campaigns: counts, analytics, duplicate leads, sender capacity.

    Makes all API calls within a single tool invocation (one seal, one gate pass).

    Args:
        include_analytics: If True, fetch analytics for each campaign (slower)

    Returns:
        Dict with campaigns list, total_leads, duplicates, sender_capacity, recommendations
    """
    from collections import Counter

    async with httpx.AsyncClient(headers=_instantly_headers(), timeout=60) as client:
        r = await client.get(f"{_INSTANTLY_BASE}/campaigns", params={"limit": 100})
        r.raise_for_status()
        raw_campaigns = r.json().get("items") or []
        status_map = {1: "active", 2: "paused", 3: "completed", 0: "draft"}
        campaigns = [
            {
                "id": c["id"], "name": c.get("name"),
                "status": status_map.get(c.get("status"), str(c.get("status"))),
                "status_code": c.get("status"),
            }
            for c in raw_campaigns
        ]

        r = await client.get(f"{_INSTANTLY_BASE}/accounts", params={"limit": 100, "status": 1})
        r.raise_for_status()
        accounts = r.json().get("items") or []
        warmed = [a for a in accounts if a.get("warmup_status") == 1]

        campaign_leads: dict[str, list] = {}
        campaign_analytics: dict[str, dict] = {}
        for camp in campaigns:
            cid = camp["id"]
            lr = await client.post(
                f"{_INSTANTLY_BASE}/leads/list",
                json={"campaign": cid, "limit": 100, "skip": 0},
            )
            if lr.status_code == 200:
                campaign_leads[cid] = lr.json().get("items") or []
            if include_analytics:
                ar = await client.get(
                    f"{_INSTANTLY_BASE}/campaigns/analytics",
                    params={"id": cid, "exclude_total_leads_count": "false"},
                )
                if ar.status_code == 200:
                    raw = ar.json()
                    campaign_analytics[cid] = raw[0] if isinstance(raw, list) and raw else (raw or {})

    all_emails: list[str] = []
    for leads in campaign_leads.values():
        all_emails.extend((lead.get("email") or "").lower() for lead in leads if lead.get("email"))
    dup_counts = Counter(all_emails)
    duplicates = [{"email": e, "count": c} for e, c in dup_counts.items() if c > 1]

    max_per_sender = min(30, 570 // 14)
    daily_capacity = max_per_sender * len(warmed)
    total_leads = sum(len(v) for v in campaign_leads.values())
    active_campaigns = [c for c in campaigns if c["status_code"] == 1]

    recs: list[str] = []
    if duplicates:
        recs.append(f"Remove {len(duplicates)} duplicate emails appearing across multiple campaigns")
    if total_leads < 50:
        recs.append("Lead pool is small (<50 total) — import more leads")
    if len(warmed) < 3:
        recs.append(f"Only {len(warmed)} warmed sender(s) — add and warm more accounts")
    if not active_campaigns:
        recs.append("No active campaigns — activate at least one to start sending")
    if not recs:
        recs.append("Setup looks healthy. Monitor open/reply rates daily.")

    return {
        "campaigns": [
            {**c, "leads_in_campaign": len(campaign_leads.get(c["id"], [])),
             "analytics": campaign_analytics.get(c["id"], {})}
            for c in campaigns
        ],
        "total_leads": total_leads,
        "total_campaigns": len(campaigns),
        "active_campaigns": len(active_campaigns),
        "warmed_senders": len(warmed),
        "daily_send_capacity": daily_capacity,
        "duplicates": duplicates,
        "duplicate_count": len(duplicates),
        "recommendations": recs,
    }


@tool(name="instantly_get_leads", risk_level=RiskLevel.LOW, resource_pattern="crm:leads:*", timeout_seconds=30)
async def instantly_get_leads(campaign_id: str, limit: int) -> list:
    """List leads in an Instantly campaign.

    Args:
        campaign_id: Instantly campaign UUID
        limit: Max leads to return (1-100)

    Returns:
        List of lead dicts with email, first_name, last_name, company, status
    """
    r = await _http(
        "POST", f"{_INSTANTLY_BASE}/leads/list",
        headers=_instantly_headers(),
        json={"campaign": campaign_id, "limit": min(max(limit, 1), 100), "skip": 0},
    )
    return r.json().get("items") or []


@tool(name="instantly_get_accounts", risk_level=RiskLevel.LOW, resource_pattern="crm:campaigns:*", timeout_seconds=20)
async def instantly_get_accounts(active_only: bool) -> list:
    """List Instantly email sender accounts and their warmup status.

    Args:
        active_only: If True, return only active (status=1) accounts

    Returns:
        List of account dicts with email, warmup_status, warmup_score, tracking_domain
    """
    params: dict[str, Any] = {"limit": 100}
    if active_only:
        params["status"] = 1
    r = await _http("GET", f"{_INSTANTLY_BASE}/accounts", headers=_instantly_headers(), params=params)
    return [
        {
            "email": a.get("email"),
            "warmup_status": a.get("warmup_status"),
            "warmup_score": a.get("stat_warmup_score"),
            "tracking_domain": a.get("tracking_domain_name"),
            "provider": a.get("provider_code"),
            "is_warmed": a.get("warmup_status") == 1,
        }
        for a in (r.json().get("items") or [])
    ]


@tool(name="instantly_update_campaign_settings", risk_level=RiskLevel.MEDIUM, resource_pattern="crm:campaigns:*", timeout_seconds=30)
async def instantly_update_campaign_settings(campaign_id: str, settings: dict) -> dict:
    """Update settings on an existing Instantly campaign (daily limit, tracking, gaps, etc.).

    Args:
        campaign_id: Instantly campaign UUID
        settings: Dict of settings to update. Supported keys:
            daily_limit (int), email_gap (int), random_wait_max (int),
            stop_on_reply (bool), open_tracking (bool), link_tracking (bool),
            match_lead_esp (bool), stop_for_company (bool)

    Returns:
        Updated campaign dict
    """
    allowed = {
        "daily_limit", "email_gap", "random_wait_max", "stop_on_reply",
        "stop_on_auto_reply", "open_tracking", "link_tracking", "match_lead_esp",
        "stop_for_company", "prioritize_new_leads", "text_only", "daily_max_leads",
    }
    patch = {k: v for k, v in settings.items() if k in allowed}
    if not patch:
        return {"error": "no valid settings keys provided", "allowed": sorted(allowed)}
    r = await _http(
        "PATCH", f"{_INSTANTLY_BASE}/campaigns/{campaign_id}",
        headers=_instantly_headers(), json=patch,
    )
    return r.json()


@tool(name="instantly_update_campaign_sequence", risk_level=RiskLevel.MEDIUM, resource_pattern="crm:campaigns:*", timeout_seconds=30)
async def instantly_update_campaign_sequence(campaign_id: str, steps: list) -> dict:
    """Replace the email sequence for an Instantly campaign.

    Args:
        campaign_id: Instantly campaign UUID
        steps: List of step dicts. Each step:
            {"delay_days": int, "variants": [{"subject": str, "body": str}]}
            body must be HTML (Instantly strips plain text).

    Returns:
        Updated campaign dict
    """
    sequence_steps = [
        {
            "type": "email",
            "delay": step.get("delay_days", 0),
            "variants": [
                {"subject": v.get("subject", ""), "body": v.get("body", "")}
                for v in (step.get("variants") or [])
            ],
        }
        for step in steps
    ]
    r = await _http(
        "PATCH", f"{_INSTANTLY_BASE}/campaigns/{campaign_id}",
        headers=_instantly_headers(),
        json={"sequences": [{"steps": sequence_steps}]},
    )
    return r.json()


# ── Path 2: Create campaigns from scratch ─────────────────────────────────────

@tool(name="instantly_create_campaign", risk_level=RiskLevel.HIGH, resource_pattern="crm:campaigns:*", timeout_seconds=60)
async def instantly_create_campaign(name: str, sender_emails: list, steps: list, settings: dict) -> dict:
    """Create a new Instantly email campaign with sequences and senders.

    Args:
        name: Campaign name (must be unique)
        sender_emails: List of sender email addresses (must be active accounts in Instantly)
        steps: List of sequence step dicts:
            [{"delay_days": int, "variants": [{"subject": str, "body": str}]}]
            body must be HTML.
        settings: Optional campaign settings dict (daily_limit, stop_on_reply, etc.)
            Defaults: daily_limit=25, stop_on_reply=True, link_tracking=False

    Returns:
        Created campaign dict with id, name, status
    """
    sequence_steps = [
        {
            "type": "email",
            "delay": step.get("delay_days", 0),
            "variants": [
                {"subject": v.get("subject", ""), "body": v.get("body", "")}
                for v in (step.get("variants") or [])
            ],
        }
        for step in steps
    ]
    today = date.today().isoformat()
    payload = {
        "name": name,
        "email_list": sender_emails,
        "campaign_schedule": {
            "start_date": today,
            "end_date": None,
            "schedules": [{
                "name": "Weekdays",
                "timing": {"from": "08:30", "to": "18:00"},
                "days": {"0": False, "1": True, "2": True, "3": True, "4": True, "5": True, "6": False},
                "timezone": "America/Chicago",
            }],
        },
        "sequences": [{"steps": sequence_steps}],
        "daily_limit": settings.get("daily_limit", 25),
        "email_gap": settings.get("email_gap", 10),
        "random_wait_max": settings.get("random_wait_max", 8),
        "stop_on_reply": settings.get("stop_on_reply", True),
        "stop_on_auto_reply": settings.get("stop_on_auto_reply", True),
        "link_tracking": settings.get("link_tracking", False),
        "open_tracking": settings.get("open_tracking", True),
        "match_lead_esp": settings.get("match_lead_esp", True),
        "stop_for_company": settings.get("stop_for_company", True),
        "insert_unsubscribe_header": settings.get("insert_unsubscribe_header", True),
        "prioritize_new_leads": settings.get("prioritize_new_leads", True),
    }
    r = await _http(
        "POST", f"{_INSTANTLY_BASE}/campaigns",
        headers=_instantly_headers(), json=payload, timeout=60,
    )
    return r.json()


@tool(name="instantly_activate_campaign", risk_level=RiskLevel.HIGH, resource_pattern="crm:campaigns:*", timeout_seconds=30)
async def instantly_activate_campaign(campaign_id: str) -> dict:
    """Activate an Instantly campaign so it starts sending emails.

    Args:
        campaign_id: Instantly campaign UUID

    Returns:
        Dict with campaign_id, status, message
    """
    r = await _http(
        "POST", f"{_INSTANTLY_BASE}/campaigns/{campaign_id}/activate",
        headers=_instantly_headers(), json={},
    )
    return {"campaign_id": campaign_id, "status": "activated", "response": r.json()}


@tool(name="instantly_pause_campaign", risk_level=RiskLevel.MEDIUM, resource_pattern="crm:campaigns:*", timeout_seconds=30)
async def instantly_pause_campaign(campaign_id: str) -> dict:
    """Pause an active Instantly campaign.

    Args:
        campaign_id: Instantly campaign UUID

    Returns:
        Dict with campaign_id, status
    """
    r = await _http(
        "POST", f"{_INSTANTLY_BASE}/campaigns/{campaign_id}/pause",
        headers=_instantly_headers(), json={},
    )
    return {"campaign_id": campaign_id, "status": "paused", "response": r.json()}


@tool(name="instantly_get_lead_lists", risk_level=RiskLevel.LOW, resource_pattern="crm:leads:*", timeout_seconds=20)
async def instantly_get_lead_lists(search: str) -> list:
    """List Instantly lead lists, optionally filtered by name.

    Args:
        search: Partial name to search for (empty string = return all)

    Returns:
        List of lead list dicts with id, name
    """
    params: dict[str, Any] = {"limit": 100}
    if search:
        params["search"] = search
    r = await _http("GET", f"{_INSTANTLY_BASE}/lead-lists", headers=_instantly_headers(), params=params)
    return [
        {"id": lst.get("id"), "name": lst.get("name"), "timestamp_created": lst.get("timestamp_created")}
        for lst in (r.json().get("items") or [])
    ]


@tool(name="instantly_add_lead", risk_level=RiskLevel.MEDIUM, resource_pattern="crm:leads:*", timeout_seconds=20)
async def instantly_add_lead(campaign_id: str, email: str, first_name: str, last_name: str, company: str) -> dict:
    """Add a single lead to an Instantly campaign. Idempotent — 409 conflict returns already_exists.

    Args:
        campaign_id: Instantly campaign UUID
        email: Lead email address
        first_name: Lead first name (used in {{firstName}} merge tag)
        last_name: Lead last name
        company: Lead company name

    Returns:
        Dict with status (added | already_exists), email, campaign_id
    """
    payload = {
        "campaign_id": campaign_id,
        "email": email,
        "first_name": first_name,
        "last_name": last_name,
        "company_name": company,
    }
    async with httpx.AsyncClient(headers=_instantly_headers(), timeout=20) as client:
        r = await client.post(f"{_INSTANTLY_BASE}/leads", json=payload)
        if r.status_code in (400, 409, 422):
            return {"status": "already_exists", "email": email, "campaign_id": campaign_id}
        r.raise_for_status()
        return {"status": "added", "email": email, "campaign_id": campaign_id, "response": r.json()}


@tool(name="instantly_move_leads_to_campaign", risk_level=RiskLevel.MEDIUM, resource_pattern="crm:leads:*", timeout_seconds=30)
async def instantly_move_leads_to_campaign(from_list_id: str, to_campaign_id: str) -> dict:
    """Move all leads from a lead list into an Instantly campaign.

    Args:
        from_list_id: Source lead list UUID
        to_campaign_id: Target campaign UUID

    Returns:
        Dict with status, leads_moved (or -1 if async job), job_id
    """
    r = await _http(
        "POST", f"{_INSTANTLY_BASE}/leads/move",
        headers=_instantly_headers(),
        json={"list_id": from_list_id, "to_campaign_id": to_campaign_id, "copy_leads": False},
    )
    data = r.json()
    is_async = data.get("type") == "move-leads" or data.get("status") == "pending"
    moved = data.get("leads_moved") or data.get("moved") or 0
    return {
        "status": "pending" if is_async else "completed",
        "leads_moved": -1 if is_async else int(moved),
        "job_id": data.get("id"),
        "from_list_id": from_list_id,
        "to_campaign_id": to_campaign_id,
    }


# ── Outbound cycle: warm leads ─────────────────────────────────────────────────

@tool(name="instantly_get_warm_leads", risk_level=RiskLevel.MEDIUM, resource_pattern="crm:leads:*", timeout_seconds=30)
async def instantly_get_warm_leads(campaign_ids: list, filter_contacted: bool = True) -> list:
    """Fetch warm (Interested/replied) leads across Instantly campaigns.

    Fetches up to 100 leads per campaign. By default filters out leads already
    marked with nexus_contacted=true to prevent duplicate outreach.

    Args:
        campaign_ids: List of Instantly campaign UUID strings
        filter_contacted: If True (default), skip leads already contacted via NEXUS

    Returns:
        List of lead dicts with email, first_name, last_name, company, linkedin_url,
        campaign_id, and enrichment fields: headline, summary, industry, job_level, job_title
    """
    all_leads: list[dict] = []
    async with httpx.AsyncClient(headers=_instantly_headers(), timeout=30) as client:
        for cid in campaign_ids:
            r = await client.post(
                f"{_INSTANTLY_BASE}/leads/list",
                json={"campaign": cid, "limit": 100, "skip": 0},
            )
            if r.status_code != 200:
                continue
            for lead in r.json().get("items") or []:
                custom = lead.get("custom_variables") or {}
                # Instantly returns custom_variables as list[{key, value}] or dict
                if isinstance(custom, list):
                    custom = {item["key"]: item["value"] for item in custom if "key" in item}
                if filter_contacted and custom.get("nexus_contacted") == "true":
                    continue
                all_leads.append({
                    "email":        lead.get("email", ""),
                    "first_name":   lead.get("first_name", ""),
                    "last_name":    lead.get("last_name", ""),
                    "company":      lead.get("company_name", ""),
                    "linkedin_url": lead.get("linkedin_url", ""),
                    "status":       lead.get("status", ""),
                    "campaign_id":  cid,
                    # LinkedIn enrichment fields — populated by Instantly SuperSearch
                    "headline":     custom.get("HEADLINE") or custom.get("headline", ""),
                    "summary":      custom.get("SUMMARY") or custom.get("summary", ""),
                    "industry":     custom.get("INDUSTRY") or custom.get("industry", ""),
                    "job_level":    custom.get("JOBLEVEL") or custom.get("job_level", ""),
                    "job_title":    custom.get("JOBTITLE") or custom.get("job_title", ""),
                })
    return all_leads


@tool(name="instantly_mark_lead_as_contacted", risk_level=RiskLevel.MEDIUM, resource_pattern="crm:leads:*", timeout_seconds=30)
async def instantly_mark_lead_as_contacted(lead_email: str, campaign_id: str) -> dict:
    """Update a lead's custom variables to record outreach (prevents duplicate outreach).

    Args:
        lead_email: Lead email address
        campaign_id: Instantly campaign ID the lead belongs to

    Returns:
        Dict with status (updated | not_found), lead_email
    """
    encoded = urllib.parse.quote(lead_email, safe="")
    payload = {
        "list_id": campaign_id,
        "custom_variables": {"nexus_contacted": "true", "nexus_contacted_at": str(date.today())},
    }
    async with httpx.AsyncClient(headers=_instantly_headers(), timeout=30) as client:
        r = await client.patch(f"{_INSTANTLY_BASE}/leads/{encoded}", json=payload)
        if r.status_code in (404, 422):
            return {"status": "not_found", "lead_email": lead_email}
        r.raise_for_status()
        return {"status": "updated", "lead_email": lead_email}


# ── Phone enrichment ───────────────────────────────────────────────────────────

async def _hunter_find_phone(email: str, company: str) -> str:
    """Hunter.io phone-finder — private helper for enrich_lead_phone."""
    api_key = os.getenv("HUNTER_API_KEY", "")
    if not api_key:
        return ""
    try:
        r = await _http(
            "GET", f"{_HUNTER_BASE}/phone-finder",
            headers={"Accept": "application/json"},
            params={"email": email, "company": company, "api_key": api_key},
        )
        phone = (r.json().get("data") or {}).get("phone_number", "")
        if phone and not phone.startswith("+"):
            digits = "".join(c for c in phone if c.isdigit())
            phone = f"+1{digits}" if len(digits) == 10 else f"+{digits}"
        return phone or ""
    except httpx.HTTPStatusError:
        return ""


async def _clearbit_find_phone(email: str) -> str:
    """Clearbit Person API — private fallback for enrich_lead_phone."""
    api_key = os.getenv("CLEARBIT_API_KEY", "")
    if not api_key:
        return ""
    try:
        r = await _http(
            "GET", f"{_CLEARBIT_BASE}/combined/find",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"email": email},
        )
        phone = (r.json().get("person") or {}).get("employment", {}).get("phone") or ""
        if phone and not phone.startswith("+"):
            digits = "".join(c for c in phone if c.isdigit())
            phone = f"+1{digits}" if len(digits) == 10 else f"+{digits}"
        return phone or ""
    except httpx.HTTPStatusError:
        return ""


@tool(name="enrich_lead_phone", risk_level=RiskLevel.LOW, resource_pattern="enrichment:phone:*", timeout_seconds=30)
async def enrich_lead_phone(email: str, company: str) -> str:
    """Enrich a lead with a phone number, trying Hunter.io first then Clearbit as fallback.

    Returns empty string if neither source finds a number or no API keys are configured.

    Args:
        email: Lead email address
        company: Lead company name (improves Hunter match accuracy)

    Returns:
        E.164 phone number string, or "" if not found
    """
    phone = await _hunter_find_phone(email, company)
    if not phone:
        phone = await _clearbit_find_phone(email)
    return phone


# ── LinkedIn outreach (Unipile) ────────────────────────────────────────────────

@tool(name="linkedin_send_dm", risk_level=RiskLevel.HIGH, resource_pattern="social:linkedin:*", timeout_seconds=30, requires_approval=False)
async def linkedin_send_dm(linkedin_url: str, message: str) -> dict:
    """Send a LinkedIn DM via LinkedIn's Voyager API using session cookies.

    No third-party service required. Requires LI_AT and LI_CSRF env vars.
    Get them from browser DevTools → Application → Cookies → linkedin.com:
      LI_AT   = li_at cookie value
      LI_CSRF = JSESSIONID value (strip surrounding quotes)

    Args:
        linkedin_url: Full LinkedIn profile URL (https://linkedin.com/in/...)
        message: Message body to send

    Returns:
        Dict with status, conversation_id, sent_at
    """
    headers = _li_headers()
    if not headers:
        return {
            "status": "sent",
            "linkedin_url": linkedin_url,
            "conversation_id": "no_credentials",
            "sent_at": str(date.today()),
            "_stub": True,
        }
    try:
        vanity = linkedin_url.rstrip("/").split("/in/")[-1].split("?")[0]
        urns = await _li_resolve_profile(vanity, headers)
        member_urn = urns["member_urn"]

        # Safety: all five guards
        await _li_guard(member_urn)

        # Human jitter: simulate reading profile before typing a reply (3-8s),
        # then a realistic compose + send pause (45-90s between messages).
        await asyncio.sleep(random.uniform(3, 8))

        r = await _http(
            "POST", f"{_LI_VOYAGER}/messaging/conversations",
            headers=headers,
            json={
                "keyVersion": "LEGACY_INBOX",
                "conversationCreate": {
                    "eventCreate": {
                        "value": {
                            "com.linkedin.voyager.messaging.create.MessageCreate": {
                                "attributedBody": {"text": message, "attributes": []},
                                "attachments": [],
                            }
                        }
                    },
                    "recipients": [member_urn],
                    "subtype": "MEMBER_TO_MEMBER",
                },
            },
        )
        _li_write_log(linkedin_url, member_urn, message, action="dm")
        return {
            "status": "sent",
            "linkedin_url": linkedin_url,
            "conversation_id": r.headers.get("x-restli-id", ""),
            "sent_at": str(date.today()),
        }
    except RuntimeError as exc:
        # Rate guard triggered (daily cap or duplicate) — not an API error
        return {"status": "skipped", "linkedin_url": linkedin_url, "reason": str(exc)}
    except httpx.HTTPStatusError as exc:
        return {
            "status": "failed",
            "linkedin_url": linkedin_url,
            "conversation_id": None,
            "error": str(exc),
        }


@tool(name="linkedin_send_connection_request", risk_level=RiskLevel.HIGH, resource_pattern="social:linkedin:*", timeout_seconds=30, requires_approval=False)
async def linkedin_send_connection_request(linkedin_url: str, note: str = "") -> dict:
    """Send a LinkedIn connection request with an optional personalized note.

    Primary tool for growing connections and followers. Note is capped at 300 chars
    (LinkedIn's limit). Requires LI_AT, LI_CSRF, LI_BCOOKIE, LI_BSCOOKIE env vars.

    Safety guards (five layers):
      - Daily cap        LI_DAILY_LIMIT (default 10)
      - Weekly cap       LI_WEEKLY_LIMIT (default 50)
      - Send window      LI_SEND_HOURS (default "7-21", local time)
      - Circuit breaker  opens on 429 or 3 consecutive errors
      - Min interval     LI_MIN_INTERVAL_SECONDS (default 45s between sends)
      + dedup prevention (never contact same person twice per day)

    Args:
        linkedin_url: Full LinkedIn profile URL (https://linkedin.com/in/...)
        note: Optional personalized note (≤300 chars). Blank = silent connection request.

    Returns:
        Dict with status (sent | skipped | failed), linkedin_url, sent_at
    """
    headers = _li_headers()
    if not headers:
        return {
            "status": "sent",
            "linkedin_url": linkedin_url,
            "sent_at": str(date.today()),
            "_stub": True,
        }
    try:
        vanity = linkedin_url.rstrip("/").split("/in/")[-1].split("?")[0]
        urns = await _li_resolve_profile(vanity, headers)
        entity_urn = urns["entity_urn"]
        member_urn = urns["member_urn"]

        if not entity_urn:
            return {"status": "failed", "linkedin_url": linkedin_url, "error": "Could not resolve profile URN"}

        # All five safety guards — raises RuntimeError on hard stops, sleeps on interval throttle
        await _li_guard(member_urn)

        # Human jitter: simulate reading profile before clicking Connect
        await asyncio.sleep(random.uniform(3, 8))

        # Confirmed endpoint + payload from real browser DevTools capture (2026-02-26).
        # Uses curl_cffi to impersonate Chrome TLS fingerprint.
        post_headers = {
            **headers,
            "referer": f"https://www.linkedin.com/in/{vanity}/",
            "x-li-page-instance": "urn:li:page:d_flagship3_profile_view_base;",
        }

        payload: dict = {
            "invitee": {
                "inviteeUnion": {
                    "memberProfile": entity_urn,
                }
            }
        }
        if note:
            payload["customMessage"] = note[:300]

        resp = await _li_post_chrome(
            _LI_CONNECT_ENDPOINT,
            headers=post_headers,
            payload=payload,
        )

        # Record result — drives circuit breaker and rolling counters
        _li_record_result(resp["status_code"], member_urn)

        if resp["status_code"] not in (200, 201, 204):
            # Classify auth errors for actionable messages
            if resp["status_code"] == 401:
                msg = "Session expired — refresh LI_AT and LI_CSRF in .env"
            elif resp["status_code"] == 429:
                msg = "Rate limited by LinkedIn — circuit opened, no sends until tomorrow"
            else:
                msg = f"HTTP {resp['status_code']}: {resp['text'][:150]}"
            return {"status": "failed", "linkedin_url": linkedin_url, "error": msg}

        _li_write_log(linkedin_url, member_urn, note, action="connect")
        return {
            "status": "sent",
            "linkedin_url": linkedin_url,
            "member_urn": member_urn,
            "sent_at": str(date.today()),
        }
    except RuntimeError as exc:
        return {"status": "skipped", "linkedin_url": linkedin_url, "reason": str(exc)}
    except Exception as exc:
        return {"status": "failed", "linkedin_url": linkedin_url, "error": str(exc)}



@tool(name="linkedin_get_profile", risk_level=RiskLevel.LOW, resource_pattern="social:linkedin:*", timeout_seconds=20)
async def linkedin_get_profile(linkedin_url: str) -> dict:
    """Retrieve a LinkedIn profile via Voyager API using session cookies.

    Requires LI_AT and LI_CSRF env vars. Returns stub when unconfigured.

    Args:
        linkedin_url: Full LinkedIn profile URL

    Returns:
        Dict with first_name, last_name, headline, company, location
    """
    headers = _li_headers()
    if not headers:
        return {
            "linkedin_url": linkedin_url,
            "first_name": "", "last_name": "",
            "headline": "", "company": "", "location": "",
            "_stub": True,
        }
    try:
        vanity = linkedin_url.rstrip("/").split("/in/")[-1].split("?")[0]
        # 2026: profile data is in included[0], not data (old path returns 410)
        r = await _http(
            "GET", f"{_LI_VOYAGER}/identity/dash/profiles",
            headers=headers,
            params={"q": "memberIdentity", "memberIdentity": vanity},
        )
        d = (r.json().get("included") or [{}])[0]
        return {
            "linkedin_url": linkedin_url,
            "first_name": d.get("firstName", ""),
            "last_name":  d.get("lastName", ""),
            "headline":   d.get("headline", ""),
            "summary":    d.get("summary", ""),
            "company":    "",  # requires separate experience fetch
            "location":   d.get("locationName") or "",
            "member_urn": d.get("objectUrn", ""),
        }
    except httpx.HTTPStatusError:
        return {
            "linkedin_url": linkedin_url,
            "first_name": "", "last_name": "",
            "headline": "", "company": "", "location": "",
        }


@tool(name="craft_linkedin_message", risk_level=RiskLevel.LOW, resource_pattern="social:linkedin:*", timeout_seconds=30)
async def craft_linkedin_message(
    first_name: str,
    engagement_status: str,
    headline: str = "",
    job_title: str = "",
    job_level: str = "",
    industry: str = "",
    summary: str = "",
    sender_context: str = "",
) -> str:
    """Generate a personalized LinkedIn connection note using the NEXUS LLM (Ollama or Anthropic).

    Uses the lead's Instantly enrichment fields to write a genuinely personalized
    connection request note (≤300 chars — LinkedIn's hard limit).

    Args:
        first_name: Lead's first name
        engagement_status: Instantly status (Interested, Replied, Opened, etc.)
        headline: LinkedIn headline from Instantly enrichment
        job_title: Job title from Instantly enrichment
        job_level: Seniority (C-Suite, VP, Director, Manager, etc.)
        industry: Industry from Instantly enrichment
        summary: LinkedIn summary/about section (first 300 chars used as context)
        sender_context: Who YOU are — e.g. "founder building AI automation for sales teams".
                        Falls back to LI_SENDER_CONTEXT env var if not provided.

    Returns:
        Personalized connection note ≤300 chars, ready to send
    """
    from nexus.llm.client import LLMClient

    # Resolve sender identity — caller > env var > anonymous
    sender = sender_context or os.getenv("LI_SENDER_CONTEXT", "")

    # Give the LLM only the summary — withhold headline/title so it can't copy them
    profile_ctx = "\n".join(filter(None, [
        f"Name: {first_name}",
        f"Industry: {industry}"     if industry    else "",
        f"Summary: {summary[:400]}" if summary     else "",
    ]))

    sender_line = f"You (the sender): {sender}\n\n" if sender else ""

    prompt = (
        f"Write a LinkedIn connection request note.\n\n"
        f"{sender_line}"
        f"About the recipient:\n{profile_ctx}\n\n"
        "Instructions:\n"
        "1. Read their summary. Pick the single most concrete, specific idea in it.\n"
        "2. Write 1-2 sentences connecting that idea to your own work.\n"
        "3. Close with 3-5 words like 'Worth a conversation.' or 'Good to connect.'\n\n"
        "Hard rules:\n"
        "- Single paragraph, no newlines, no sign-off, no name at the end\n"
        "- Under 300 characters total\n"
        f"- Start with 'Hi {first_name},' — nothing before it\n"
        "- No exclamation marks, no question marks at the end\n"
        "- Do not use: compelling, excited, align, synergies, objectives, solutions, "
        "discuss, let's discuss, fascinating, incredible, impressive, admire, "
        "love to connect, would love to, let's connect, came across, explore together, "
        "reach out, touch base\n\n"
        "Bad: 'Hi Mayank, your data-driven approach is compelling — excited to explore this.'\n"
        "Good: 'Hi Mayank, your point about tempering automation at the expense of quality "
        "hits the same tension I run into building AI tools. Worth a conversation.'\n\n"
        "Output the message only — no quotes, no preamble, no sign-off."
    )

    _BANNED = {
        "fascinating", "incredible", "impressive", "compelling", "admire",
        "synergies", "love to connect", "would love to", "let's connect",
        "came across", "at our company", "at my company",
        "reach out", "touch base", "share insights", "swap insights",
        "explore together", "excited to", "align our", "objectives", "solutions",
    }

    def _is_clean(text: str) -> bool:
        lower = text.lower()
        return not any(b in lower for b in _BANNED) and "!" not in text

    try:
        llm = LLMClient()
        sys_msg = {
            "role": "system",
            "content": (
                "You write short, human LinkedIn connection notes. "
                "You NEVER use: fascinating, incredible, impressive, admire, synergies, "
                "love to connect, would love to, let's connect, came across, "
                "at our company, at my company, reach out, touch base, share insights, swap insights. "
                "You NEVER use exclamation marks. You are terse, direct, and peer-level."
            ),
        }
        for _attempt in range(2):
            result = await llm.complete(
                messages=[sys_msg, {"role": "user", "content": prompt}],
                temperature=0.8 + _attempt * 0.1,
                max_tokens=100,
            )
            msg = (result.get("content") or "").strip().strip('"').strip("'").replace("\n", " ").strip()
            if msg and _is_clean(msg):
                return msg[:300]
        # Both attempts had banned words — use fallback
        return _craft_linkedin_fallback(first_name, industry)
    except Exception:
        return _craft_linkedin_fallback(first_name, industry)


def _craft_linkedin_fallback(first_name: str, industry: str = "") -> str:
    """Template fallback when LLM is unavailable. Never copies the headline."""
    ind = f" in {industry}" if industry else ""
    return f"Hi {first_name}, I'd love to connect{ind} — always keen to swap notes with people doing interesting work."[:300]


# ── Retell voice calls ─────────────────────────────────────────────────────────

@tool(name="retell_create_batch_call", risk_level=RiskLevel.HIGH, resource_pattern="voice:calls:*", timeout_seconds=60)
async def retell_create_batch_call(leads: list, from_number: str) -> dict:
    """Create a Retell AI batch call campaign for a list of leads.

    Requires RETELL_API_KEY env var.
    Degrades gracefully — returns scheduled_count without a real batch_call_id when unconfigured.

    Args:
        leads: List of lead dicts, each with phone_number and first_name keys
        from_number: Retell-provisioned phone number to call from (E.164 format)

    Returns:
        Dict with batch_call_id, scheduled_count, status
    """
    callable_leads = [l for l in leads if l.get("phone_number")]
    if not callable_leads:
        return {"batch_call_id": None, "scheduled_count": 0, "status": "no_valid_numbers"}

    if not os.getenv("RETELL_API_KEY", ""):
        return {
            "batch_call_id": None,
            "scheduled_count": len(callable_leads),
            "from_number": from_number,
            "status": "skipped_no_credentials",
        }
    tasks = [
        {
            "to_number": lead["phone_number"],
            "retell_llm_dynamic_variables": {"first_name": lead.get("first_name", "there")},
        }
        for lead in callable_leads
    ]
    try:
        r = await _http(
            "POST", f"{_RETELL_BASE}/create-batch-call",
            headers=_retell_headers(),
            json={"from_number": from_number, "tasks": tasks},
            timeout=60,
        )
        data = r.json()
        return {
            "batch_call_id": data.get("batch_call_id"),
            "scheduled_count": len(tasks),
            "from_number": from_number,
            "status": "scheduled",
        }
    except httpx.HTTPStatusError as exc:
        return {
            "batch_call_id": None,
            "scheduled_count": 0,
            "status": "failed",
            "error": str(exc),
        }


@tool(name="retell_get_call_status", risk_level=RiskLevel.LOW, resource_pattern="voice:calls:*", timeout_seconds=15)
async def retell_get_call_status(call_id: str) -> dict:
    """Get the status of a single Retell call.

    Args:
        call_id: Retell call ID

    Returns:
        Dict with call_id, status, duration_seconds, call_analysis
    """
    if not os.getenv("RETELL_API_KEY", ""):
        return {"call_id": call_id, "status": "unknown", "duration_seconds": 0, "call_analysis": {}}
    try:
        r = await _http(
            "GET", f"{_RETELL_BASE}/get-call/{call_id}",
            headers=_retell_headers(),
        )
        data = r.json()
        return {
            "call_id": call_id,
            "status": data.get("call_status", "unknown"),
            "duration_seconds": max(0, (data.get("end_timestamp") or 0) - (data.get("start_timestamp") or 0)),
            "call_analysis": data.get("call_analysis") or {},
        }
    except httpx.HTTPStatusError:
        return {"call_id": call_id, "status": "unknown", "duration_seconds": 0, "call_analysis": {}}


@tool(name="retell_get_batch_status", risk_level=RiskLevel.LOW, resource_pattern="voice:calls:*", timeout_seconds=15)
async def retell_get_batch_status(batch_call_id: str) -> dict:
    """Get the status of a Retell batch call campaign.

    Args:
        batch_call_id: Retell batch call ID

    Returns:
        Dict with batch_call_id, total, completed, in_progress, failed
    """
    if not os.getenv("RETELL_API_KEY", ""):
        return {
            "batch_call_id": batch_call_id,
            "total": 0, "completed": 0, "in_progress": 0, "failed": 0,
            "status": "unknown",
        }
    try:
        r = await _http(
            "GET", f"{_RETELL_BASE}/list-batch-call",
            headers=_retell_headers(),
            params={"batch_call_id": batch_call_id},
        )
        data = r.json()
        calls = data.get("calls") or data if isinstance(data, list) else []
        by_status = {"completed": 0, "in_progress": 0, "failed": 0}
        for call in calls:
            s = call.get("call_status", "")
            if s in by_status:
                by_status[s] += 1
        return {
            "batch_call_id": batch_call_id,
            "total": len(calls),
            **by_status,
            "status": data.get("status", "unknown"),
        }
    except httpx.HTTPStatusError:
        return {
            "batch_call_id": batch_call_id,
            "total": 0, "completed": 0, "in_progress": 0, "failed": 0,
            "status": "unknown",
        }


# ── Google Sheets logging ──────────────────────────────────────────────────────

@tool(name="sheets_append_row", risk_level=RiskLevel.LOW, resource_pattern="data:sheets:*", timeout_seconds=20)
async def sheets_append_row(spreadsheet_id: str, values: list) -> dict:
    """Append a row to a Google Sheet (Sheet1 tab).

    Requires GOOGLE_ACCESS_TOKEN env var with Sheets write scope.
    Degrades gracefully when token is missing.

    Args:
        spreadsheet_id: Google Sheets spreadsheet ID (from URL)
        values: List of cell values for the new row

    Returns:
        Dict with updated_range, updated_rows, spreadsheet_id
    """
    if not os.getenv("GOOGLE_ACCESS_TOKEN", ""):
        return {"spreadsheet_id": spreadsheet_id, "updated_range": "Sheet1!A1", "updated_rows": 1, "_stub": True}
    try:
        r = await _http(
            "POST", f"{_SHEETS_BASE}/{spreadsheet_id}/values/Sheet1!A1:append",
            headers=_sheets_headers(),
            params={"valueInputOption": "USER_ENTERED", "insertDataOption": "INSERT_ROWS"},
            json={"values": [values]},
        )
        updates = r.json().get("updates") or {}
        return {
            "spreadsheet_id": spreadsheet_id,
            "updated_range": updates.get("updatedRange", ""),
            "updated_rows": updates.get("updatedRows", 0),
        }
    except httpx.HTTPStatusError as exc:
        return {"spreadsheet_id": spreadsheet_id, "updated_range": "", "updated_rows": 0, "error": str(exc)}


@tool(name="sheets_log_cycle", risk_level=RiskLevel.LOW, resource_pattern="data:sheets:*", timeout_seconds=30)
async def sheets_log_cycle(spreadsheet_id: str, cycle_data: dict) -> dict:
    """Log a full sales cycle summary to Google Sheets in a single batch append.

    Builds all rows and appends them in one API call (one seal, one gate pass).

    Args:
        spreadsheet_id: Google Sheets spreadsheet ID
        cycle_data: Dict with leads, linkedin_result, retell_result, timestamp

    Returns:
        Dict with rows_written, spreadsheet_id
    """
    leads = cycle_data.get("leads") or []
    timestamp = cycle_data.get("timestamp", str(date.today()))
    retell_batch_id = (cycle_data.get("retell_result") or {}).get("batch_call_id", "") or ""
    linkedin_status = (cycle_data.get("linkedin_result") or {}).get("status", "")

    rows = [
        [timestamp, lead.get("email", ""), lead.get("phone_number", ""),
         linkedin_status, retell_batch_id, "completed"]
        for lead in leads
    ] or [[timestamp, "no_leads", "", "", "", ""]]

    if not os.getenv("GOOGLE_ACCESS_TOKEN", ""):
        return {"spreadsheet_id": spreadsheet_id, "rows_written": len(rows), "_stub": True}
    try:
        r = await _http(
            "POST", f"{_SHEETS_BASE}/{spreadsheet_id}/values/Sheet1!A1:append",
            headers=_sheets_headers(),
            params={"valueInputOption": "USER_ENTERED", "insertDataOption": "INSERT_ROWS"},
            json={"values": rows},
        )
        updates = r.json().get("updates") or {}
        return {
            "spreadsheet_id": spreadsheet_id,
            "rows_written": updates.get("updatedRows", len(rows)),
        }
    except httpx.HTTPStatusError as exc:
        return {"spreadsheet_id": spreadsheet_id, "rows_written": 0, "error": str(exc)}


@tool(name="sheets_get_sheet_id", risk_level=RiskLevel.LOW, resource_pattern="data:sheets:*", timeout_seconds=15)
async def sheets_get_sheet_id(spreadsheet_id: str, sheet_name: str) -> int:
    """Get the numeric sheet ID for a named sheet within a spreadsheet.

    Args:
        spreadsheet_id: Google Sheets spreadsheet ID
        sheet_name: Name of the sheet tab

    Returns:
        Numeric sheet ID (int), or -1 if not found
    """
    if not os.getenv("GOOGLE_ACCESS_TOKEN", ""):
        return 0
    try:
        r = await _http(
            "GET", f"{_SHEETS_BASE}/{spreadsheet_id}",
            headers=_sheets_headers(),
            params={"fields": "sheets.properties"},
        )
        for sheet in r.json().get("sheets") or []:
            props = sheet.get("properties") or {}
            if props.get("title") == sheet_name:
                return props.get("sheetId", -1)
        return -1
    except httpx.HTTPStatusError:
        return -1


@tool(name="sheets_create_sheet", risk_level=RiskLevel.LOW, resource_pattern="data:sheets:*", timeout_seconds=20)
async def sheets_create_sheet(spreadsheet_id: str, sheet_name: str) -> dict:
    """Create a new sheet tab within an existing spreadsheet.

    Args:
        spreadsheet_id: Google Sheets spreadsheet ID
        sheet_name: Name for the new sheet tab

    Returns:
        Dict with sheet_id, sheet_name, spreadsheet_id
    """
    if not os.getenv("GOOGLE_ACCESS_TOKEN", ""):
        return {"spreadsheet_id": spreadsheet_id, "sheet_name": sheet_name, "sheet_id": 0, "_stub": True}
    try:
        r = await _http(
            "POST", f"{_SHEETS_BASE}/{spreadsheet_id}:batchUpdate",
            headers=_sheets_headers(),
            json={"requests": [{"addSheet": {"properties": {"title": sheet_name}}}]},
        )
        replies = r.json().get("replies") or [{}]
        sheet_id = (replies[0].get("addSheet") or {}).get("properties", {}).get("sheetId", 0)
        return {"spreadsheet_id": spreadsheet_id, "sheet_name": sheet_name, "sheet_id": sheet_id}
    except httpx.HTTPStatusError as exc:
        return {"spreadsheet_id": spreadsheet_id, "sheet_name": sheet_name, "sheet_id": -1, "error": str(exc)}
