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
    flagged_for_review: bool = False
    email: str = ""
    phone: str = ""
    branch: str = ""
    member_since: str = ""


def _seed() -> dict:
    return {
        "M-1001": Member(
            id="M-1001", name="Alice Chen", status="active", balance=3000.00,
            email="alice.chen@example.com", phone="555-0101",
            branch="Downtown Branch", member_since="2018-03-11",
        ),
        "M-1002": Member(
            id="M-1002", name="Bob Ruiz", status="restricted", balance=5000.00,
            email="bob.ruiz@example.com", phone="555-0102",
            branch="Airport Branch", member_since="2015-07-22",
        ),
        "M-1003": Member(
            id="M-1003", name="Carla Diaz", status="active", balance=2000.00,
            slow=True,
            email="carla.diaz@example.com", phone="555-0103",
            branch="Main Branch", member_since="2020-01-05",
        ),
        "M-1098": Member(
            id="M-1098", name="Session Test", status="active", balance=1000.00,
            force_timeout=True,
            email="session.test@example.com", phone="555-0198",
            branch="Main Branch", member_since="2022-09-30",
        ),
        "M-1005": Member(
            id="M-1005", name="Derek Kim", status="active", balance=4000.00,
            sub_accounts=[
                SubAccount(name="Vacation Fund", account_type="savings", balance=500.0),
                SubAccount(name="Emergency Fund", account_type="savings", balance=1200.0),
                SubAccount(name="Car Fund", account_type="savings", balance=300.0),
            ],
            email="derek.kim@example.com", phone="555-0105",
            branch="Downtown Branch", member_since="2012-11-18",
        ),
        "M-1006": Member(
            id="M-1006", name="Elena Fox", status="active", balance=6000.00,
            similar_account_exists=True,
            email="elena.fox@example.com", phone="555-0106",
            branch="Main Branch", member_since="2019-04-02",
        ),
        "M-1007": Member(
            id="M-1007", name="Frank Osei", status="active", balance=100000.00,
            email="frank.osei@example.com", phone="555-0107",
            branch="Airport Branch", member_since="2009-06-14",
        ),
        "M-1099": Member(
            id="M-1099", name="Broken Page Test", status="active", balance=1500.00,
            broken=True,
            email="broken.test@example.com", phone="555-0199",
            branch="Main Branch", member_since="2021-02-27",
        ),
        "M-1010": Member(
            id="M-1010", name="Grace Nolan", status="active", balance=8000.00,
            flagged_for_review=True,
            email="grace.nolan@example.com", phone="555-0110",
            branch="Main Branch", member_since="2017-08-09",
        ),
    }


MEMBERS = _seed()


def get_member(member_id: str) -> Member | None:
    return MEMBERS.get(member_id)
