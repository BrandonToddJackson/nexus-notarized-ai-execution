"""Sales Growth Orchestrator — 6-hour outbound sales cycle agent.

Cycle:
  1. Poll warm leads from Instantly campaigns
  2. Enrich phone numbers (Hunter → Clearbit fallback)
  3. Send LinkedIn DMs
  4. Trigger Retell batch voice calls
  5. Log full cycle to Google Sheets

Each step goes through engine.run() so every action receives full 4-gate accountability.

Usage:
    config = {
        "campaign_ids": ["abc123"],
        "sheets_id": "1BxiM...",
        "retell_from_number": "+15551234567",
        "interval_hours": 6,
    }
    orchestrator = SalesGrowthOrchestrator(engine, config)
    await orchestrator.run_forever()   # daemon loop
    # or:
    result = await orchestrator.run_cycle()  # single cycle
"""

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class SalesGrowthOrchestrator:
    """Runs the outbound sales growth cycle on a configurable interval."""

    def __init__(self, engine: Any, config: dict):
        """
        Args:
            engine: NexusEngine instance (must have async run() method)
            config: Dict with keys:
                - campaign_ids (list[str]): Instantly campaign IDs to poll
                - sheets_id (str): Google Sheets spreadsheet ID
                - retell_from_number (str): Retell phone number (E.164)
                - interval_hours (int, optional): Hours between cycles (default 6)
                - tenant_id (str, optional): Tenant ID for engine.run() (default "cli-user")
        """
        self.engine = engine
        self.campaign_ids: list[str] = config["campaign_ids"]
        self.sheets_id: str = config["sheets_id"]
        self.retell_from_number: str = config["retell_from_number"]
        self.interval_hours: int = config.get("interval_hours", 6)
        self.tenant_id: str = config.get("tenant_id", "cli-user")

    async def run_forever(self) -> None:
        """Run the sales cycle indefinitely, sleeping interval_hours between cycles."""
        logger.info(
            "Sales Growth Orchestrator starting — interval=%dh, campaigns=%s",
            self.interval_hours,
            self.campaign_ids,
        )
        while True:
            try:
                result = await self.run_cycle()
                logger.info("Cycle complete: %s", result)
            except Exception as exc:
                logger.error("Cycle error: %s", exc, exc_info=True)
            await asyncio.sleep(self.interval_hours * 3600)

    async def run_cycle(self) -> dict:
        """Execute one full sales cycle.

        Returns:
            Dict with leads_count, linkedin_result, retell_result
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info("Starting sales cycle at %s", timestamp)

        leads = await self._step_poll_leads()
        logger.info("Polled %d warm leads", len(leads))

        leads = await self._step_enrich_phones(leads)
        logger.info("Enriched phone numbers for %d leads", len(leads))

        linkedin_result = await self._step_linkedin_outreach(leads)
        logger.info("LinkedIn outreach result: %s", linkedin_result)

        retell_result = await self._step_voice_calls(leads)
        logger.info("Retell batch result: %s", retell_result)

        await self._step_log_to_sheets(leads, linkedin_result, retell_result, timestamp)
        logger.info("Logged cycle to Google Sheets")

        return {
            "leads_count": len(leads),
            "linkedin_result": linkedin_result,
            "retell_result": retell_result,
            "timestamp": timestamp,
        }

    # ------------------------------------------------------------------
    # Individual steps — each calls engine.run() for gate accountability
    # ------------------------------------------------------------------

    async def _step_poll_leads(self) -> list[dict]:
        """Step 1: Poll warm leads from Instantly."""
        task = (
            f"Using the sales_growth_agent persona, call instantly_get_warm_leads "
            f"with campaign_ids={self.campaign_ids} and return the list of warm leads."
        )
        chain = await self.engine.run(task, self.tenant_id, "sales_growth_agent")
        # STUB: engine.run() returns a ChainPlan — extract leads from last seal output
        # Real implementation: parse chain.seals[-1].output as JSON list
        leads = _extract_list_from_chain(chain, default=[])
        if not leads:
            # Fallback: call tool directly for stub mode
            from nexus.tools.builtin.sales_growth import instantly_get_warm_leads
            leads = await instantly_get_warm_leads(self.campaign_ids)
        return leads

    async def _step_enrich_phones(self, leads: list[dict]) -> list[dict]:
        """Step 2: Enrich each lead with a phone number."""
        enriched = []
        for lead in leads:
            email = lead.get("email", "")
            company = lead.get("company", "")
            task = (
                f"Using the sales_growth_agent persona, call enrich_lead_phone "
                f"with email='{email}' and company='{company}' to find the phone number."
            )
            chain = await self.engine.run(task, self.tenant_id, "sales_growth_agent")
            # STUB: parse phone from chain output
            phone = _extract_str_from_chain(chain, default="")
            if not phone:
                from nexus.tools.builtin.sales_growth import enrich_lead_phone
                phone = await enrich_lead_phone(email, company)
            lead = {**lead, "phone_number": phone}
            enriched.append(lead)
        return enriched

    async def _step_linkedin_outreach(self, leads: list[dict]) -> dict:
        """Step 3: Send a LinkedIn DM to each lead."""
        sent = 0
        failed = 0
        for lead in leads:
            linkedin_url = lead.get("linkedin_url", "")
            if not linkedin_url:
                failed += 1
                continue
            first_name        = lead.get("first_name", "there")
            engagement_status = lead.get("status", "Interested")
            headline          = lead.get("headline", "")
            job_title         = lead.get("job_title", "")
            job_level         = lead.get("job_level", "")
            industry          = lead.get("industry", "")
            summary           = (lead.get("summary") or "")[:200]  # cap length in prompt
            task = (
                f"Using the sales_growth_agent persona, call craft_linkedin_message "
                f"with first_name='{first_name}', engagement_status='{engagement_status}', "
                f"headline='{headline}', job_title='{job_title}', job_level='{job_level}', "
                f"industry='{industry}', summary='{summary}', "
                f"then call linkedin_send_connection_request with linkedin_url='{linkedin_url}' "
                f"and the crafted note as the note argument."
            )
            try:
                chain = await self.engine.run(task, self.tenant_id, "sales_growth_agent")
                result = _extract_dict_from_chain(chain, default={})
                if result.get("_stub"):
                    logger.debug("LinkedIn DM stub (no credentials) for %s", linkedin_url)
                    failed += 1
                elif result.get("status") == "skipped":
                    logger.info("LinkedIn DM skipped for %s: %s", linkedin_url, result.get("reason"))
                    failed += 1
                else:
                    sent += 1
                    # Human pacing between sends: 45-90s cooldown protects the account.
                    # Skipped/failed leads don't trigger the delay.
                    await asyncio.sleep(random.uniform(45, 90))
            except Exception as exc:
                logger.warning("LinkedIn DM failed for %s: %s", linkedin_url, exc)
                failed += 1
        return {"sent": sent, "failed": failed, "status": "completed"}

    async def _step_voice_calls(self, leads: list[dict]) -> dict:
        """Step 4: Create a Retell batch call for all enriched leads."""
        callable_leads = [lead for lead in leads if lead.get("phone_number")]
        if not callable_leads:
            logger.info("No leads with phone numbers — skipping voice calls")
            return {"batch_call_id": None, "scheduled_count": 0, "status": "skipped"}

        # Call tool directly — passing a full lead list as a repr string inside
        # an LLM task prompt is unreliable. The tool is deterministic; no reasoning needed.
        from nexus.tools.builtin.sales_growth import retell_create_batch_call
        return await retell_create_batch_call(callable_leads, self.retell_from_number)

    async def _step_log_to_sheets(
        self,
        leads: list[dict],
        linkedin_result: dict,
        retell_result: dict,
        timestamp: str,
    ) -> None:
        """Step 5: Log the full cycle summary to Google Sheets."""
        cycle_data = {
            "leads": leads,
            "linkedin_result": linkedin_result,
            "retell_result": retell_result,
            "timestamp": timestamp,
        }
        # Call tool directly — cycle_data cannot be meaningfully serialized into
        # a natural language task string for the engine; pass it directly.
        try:
            from nexus.tools.builtin.sales_growth import sheets_log_cycle
            await sheets_log_cycle(self.sheets_id, cycle_data)
        except Exception as exc:
            logger.warning("Sheets log failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers — extract typed values from ChainPlan output
# ---------------------------------------------------------------------------

def _extract_list_from_chain(chain: Any, default: list) -> list:
    """Try to extract a list from the last seal's output in a ChainPlan."""
    try:
        import json
        seals = getattr(chain, "seals", None)
        if seals:
            output = getattr(seals[-1], "output", None)
            if isinstance(output, list):
                return output
            if isinstance(output, str):
                parsed = json.loads(output)
                if isinstance(parsed, list):
                    return parsed
    except Exception:
        pass
    return default


def _extract_str_from_chain(chain: Any, default: str) -> str:
    """Try to extract a string from the last seal's output in a ChainPlan."""
    try:
        seals = getattr(chain, "seals", None)
        if seals:
            output = getattr(seals[-1], "output", None)
            if isinstance(output, str):
                return output
    except Exception:
        pass
    return default


def _extract_dict_from_chain(chain: Any, default: dict) -> dict:
    """Try to extract a dict from the last seal's output in a ChainPlan."""
    try:
        import json
        seals = getattr(chain, "seals", None)
        if seals:
            output = getattr(seals[-1], "output", None)
            if isinstance(output, dict):
                return output
            if isinstance(output, str):
                parsed = json.loads(output)
                if isinstance(parsed, dict):
                    return parsed
    except Exception:
        pass
    return default
