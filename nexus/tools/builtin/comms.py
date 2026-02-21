"""Built-in communication tools: email, slack, webhook."""

from nexus.tools.plugin import tool
from nexus.types import RiskLevel


@tool(name="send_email", description="Send an email", risk_level=RiskLevel.HIGH, requires_approval=True)
async def send_email(to: str, subject: str, body: str) -> str:
    """Send an email.

    HIGH risk — requires approval in production.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body text

    Returns:
        Confirmation message
    """
    # v1 stub — replace with actual email sending
    return f"Email queued to {to}: {subject}"
