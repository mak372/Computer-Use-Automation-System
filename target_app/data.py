"""In-memory data model and seed data for the simulated legacy bank app.

Each seeded member exists to deterministically trigger one specific
scenario used by the discovery run and replay demos (see REPORT.md).
M-9999 is intentionally NOT seeded — it represents the "member not found"
business outcome.
"""

from dataclasses import dataclass, field

MAX_SUB_ACCOUNTS = 3


@dataclass
class SubAccount:
    name: str
    account_type: str
    balance: float


@dataclass
class Member:
    id: str
    name: str
    status: str  # "active" | "restricted"
    balance: float = 0.0
    sub_accounts: list = field(default_factory=list)
    slow: bool = False
    force_timeout: bool = False
    similar_account_exists: bool = False
    broken: bool = False


def _seed() -> dict:
    return {
        "M-1001": Member(
            id="M-1001", name="Alice Chen", status="active", balance=3000.00,
        ),
        "M-1002": Member(
            id="M-1002", name="Bob Ruiz", status="restricted", balance=5000.00,
        ),
        "M-1003": Member(
            id="M-1003", name="Carla Diaz", status="active", balance=2000.00,
            slow=True,
        ),
        "M-1098": Member(
            id="M-1098", name="Session Test", status="active", balance=1000.00,
            force_timeout=True,
        ),
        "M-1005": Member(
            id="M-1005", name="Derek Kim", status="active", balance=4000.00,
            sub_accounts=[
                SubAccount(name="Vacation Fund", account_type="savings", balance=500.0),
                SubAccount(name="Emergency Fund", account_type="savings", balance=1200.0),
                SubAccount(name="Car Fund", account_type="savings", balance=300.0),
            ],
        ),
        "M-1006": Member(
            id="M-1006", name="Elena Fox", status="active", balance=6000.00,
            similar_account_exists=True,
        ),
        "M-1007": Member(
            id="M-1007", name="Frank Osei", status="active", balance=100000.00,
        ),
        "M-1099": Member(
            id="M-1099", name="Broken Page Test", status="active", balance=1500.00,
            broken=True,
        ),
    }


MEMBERS = _seed()


def get_member(member_id: str) -> Member | None:
    return MEMBERS.get(member_id)
