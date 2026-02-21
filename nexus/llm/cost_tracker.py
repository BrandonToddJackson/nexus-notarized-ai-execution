"""Track LLM token usage per tenant. Budget caps and alerts."""

import logging
from typing import Optional

import litellm
from litellm import ModelResponse, Usage

from nexus.types import CostRecord
from nexus.exceptions import BudgetExceeded
from nexus.config import config

logger = logging.getLogger(__name__)


class CostTracker:
    """Track LLM token usage per tenant."""

    def __init__(self, repository=None):
        """
        Args:
            repository: DB repository for persisting costs (Phase 5).
                        Can be None for in-memory tracking.
        """
        self._repository = repository
        self._tenant_costs: dict[str, float] = {}  # in-memory fallback

    async def record(
        self,
        tenant_id: str,
        chain_id: str,
        seal_id: Optional[str],
        model: str,
        usage: dict,
    ) -> CostRecord:
        """Record LLM usage and check budget.

        Uses litellm.completion_cost() for cost calculation.

        Args:
            tenant_id: Tenant to charge
            chain_id: Chain context
            seal_id: Seal context (optional)
            model: Model used
            usage: {"input_tokens": int, "output_tokens": int}

        Returns:
            CostRecord

        Raises:
            BudgetExceeded: If tenant is over 100% budget
        """
        # Calculate cost via litellm using a ModelResponse with pre-counted tokens
        try:
            _usage = Usage(
                prompt_tokens=usage["input_tokens"],
                completion_tokens=usage["output_tokens"],
                total_tokens=usage["input_tokens"] + usage["output_tokens"],
            )
            _response = ModelResponse(model=model, usage=_usage)
            cost = litellm.completion_cost(completion_response=_response, model=model)
        except Exception:
            cost = 0.0

        record = CostRecord(
            tenant_id=tenant_id,
            chain_id=chain_id,
            seal_id=seal_id,
            model=model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cost_usd=cost,
        )

        # Update cumulative in-memory cost
        self._tenant_costs[tenant_id] = self._tenant_costs.get(tenant_id, 0.0) + cost

        # Persist if repository available
        if self._repository is not None:
            try:
                await self._repository.add_cost(record)
            except Exception as exc:
                logger.warning(f"[CostTracker] Failed to persist cost record: {exc}")

        # Budget checks
        cumulative = self._tenant_costs[tenant_id]
        budget = config.default_budget_usd
        if cumulative >= budget:
            raise BudgetExceeded(
                f"Tenant {tenant_id} has exceeded budget: "
                f"${cumulative:.4f} >= ${budget:.2f}",
                details={"tenant_id": tenant_id, "cumulative_usd": cumulative, "budget_usd": budget},
            )
        if cumulative >= budget * config.budget_alert_pct:
            logging.getLogger(__name__).warning(
                "Tenant %s at %.0f%% of budget ($%.4f / $%.2f)",
                tenant_id,
                (cumulative / budget) * 100,
                cumulative,
                budget,
            )

        return record
