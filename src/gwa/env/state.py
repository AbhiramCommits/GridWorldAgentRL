"""Deterministic synthetic CRM state.

The state is a plain dataclass graph so it can be hashed, serialized, and inspected by
programmatic success checkers without any external service.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any

CATEGORIES: tuple[str, ...] = ("billing", "technical", "account", "shipping", "refund")
STATUSES: tuple[str, ...] = ("open", "pending", "escalated", "resolved")
PRIORITIES: tuple[str, ...] = ("low", "normal", "high", "urgent")

# The one correct KB remedy code for each category. A mismatch is a realistic tool error.
CATEGORY_REMEDY: dict[str, str] = {
    "billing": "REFUND_DUPLICATE_CHARGE",
    "technical": "RESET_CACHE",
    "account": "RESET_PASSWORD",
    "shipping": "RESHIP_ORDER",
    "refund": "APPROVE_REFUND",
}

CATEGORY_TITLE: dict[str, str] = {
    "billing": "Duplicate charge refund",
    "technical": "Cache reset for dashboard errors",
    "account": "Password reset procedure",
    "shipping": "Reship missing order",
    "refund": "Refund damaged item",
}

CUSTOMER_NAMES: tuple[str, ...] = (
    "Ada Lovelace",
    "Grace Hopper",
    "Alan Turing",
    "Katherine Johnson",
    "Linus Torvalds",
    "Margaret Hamilton",
    "Dennis Ritchie",
    "Barbara Liskov",
    "Ken Thompson",
    "Radia Perlman",
)

TICKET_TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "I was charged twice for invoice INV-{n}; please refund the duplicate charge.",
        "My card shows two identical payments for invoice INV-{n}.",
    ],
    "technical": [
        "The dashboard crashes with error 500 when I open the reports page.",
        "Exports fail with a cache error after the latest release.",
    ],
    "account": [
        "I cannot reset my password; the reset email never arrives.",
        "My account is locked and the password reset link is expired.",
    ],
    "shipping": [
        "My order ORD-{n} has not arrived after two weeks.",
        "The tracking for order ORD-{n} has not updated in ten days.",
    ],
    "refund": [
        "I want a refund for order ORD-{n}; the item arrived damaged.",
        "Please refund order ORD-{n}, it is not what I ordered.",
    ],
}


@dataclass
class Customer:
    customer_id: str
    name: str
    tier: str
    email: str


@dataclass
class Ticket:
    ticket_id: str
    customer_id: str
    status: str
    priority: str
    category: str
    sla_hours_remaining: int
    text: str
    remedy_code: str | None = None
    escalation_reason: str | None = None
    replies: list[str] = field(default_factory=list)
    duplicate_of: str | None = None
    update_count: int = 0


@dataclass
class KBArticle:
    article_id: str
    category: str
    remedy_code: str
    title: str
    body: str


@dataclass
class CRMState:
    seed: int
    customers: dict[str, Customer]
    tickets: dict[str, Ticket]
    articles: list[KBArticle]
    action_log: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    invalid_actions: int = 0
    finish_summary: str | None = None
    current_turn: int = 0

    # ---- helpers ---------------------------------------------------------
    def ticket(self, ticket_id: str) -> Ticket:
        return self.tickets[ticket_id]

    def open_tickets(self) -> list[Ticket]:
        return [t for t in self.tickets.values() if t.status == "open"]

    def sorted_tickets(self) -> list[Ticket]:
        return sorted(self.tickets.values(), key=lambda t: t.ticket_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "customers": {k: asdict(v) for k, v in self.customers.items()},
            "tickets": {k: asdict(v) for k, v in self.tickets.items()},
            "articles": [asdict(a) for a in self.articles],
            "action_log": self.action_log,
            "tool_calls": self.tool_calls,
            "invalid_actions": self.invalid_actions,
            "finish_summary": self.finish_summary,
        }

    def state_hash(self) -> str:
        """Stable hash of the mutable CRM contents (not of the action log)."""
        payload = {
            "customers": {k: asdict(v) for k, v in sorted(self.customers.items())},
            "tickets": {k: asdict(v) for k, v in sorted(self.tickets.items())},
            "articles": [asdict(a) for a in self.articles],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def make_state(
    seed: int,
    n_customers: int = 8,
    n_tickets: int = 20,
    n_articles: int = 10,
) -> CRMState:
    """Build a deterministic synthetic CRM from ``seed``.

    The first tickets are structured so every task family has solvable material:
    breached-SLA tickets, a duplicate pair for one customer, and a resolved ticket
    that must reject further writes. The remaining tickets are seeded random.
    """
    if n_customers < 3:
        raise ValueError("n_customers must be >= 3")
    if n_tickets < 8:
        raise ValueError("n_tickets must be >= 8")
    if n_articles < len(CATEGORIES):
        raise ValueError(f"n_articles must be >= {len(CATEGORIES)}")

    rng = random.Random(seed)
    customers: dict[str, Customer] = {}
    for i in range(n_customers):
        cid = f"C-{i + 1:03d}"
        customers[cid] = Customer(
            customer_id=cid,
            name=CUSTOMER_NAMES[i % len(CUSTOMER_NAMES)],
            tier=("free", "pro", "enterprise")[i % 3],
            email=f"user{i + 1:03d}@example.test",
        )

    articles: list[KBArticle] = []
    for i, category in enumerate(CATEGORIES):
        articles.append(
            KBArticle(
                article_id=f"KB-{i + 1:03d}",
                category=category,
                remedy_code=CATEGORY_REMEDY[category],
                title=CATEGORY_TITLE[category],
                body=(
                    f"For {category} issues, apply remedy {CATEGORY_REMEDY[category]} "
                    f"and then notify the customer. Never apply a remedy from another category."
                ),
            )
        )
    for i in range(len(CATEGORIES), n_articles):
        category = CATEGORIES[i % len(CATEGORIES)]
        articles.append(
            KBArticle(
                article_id=f"KB-{i + 1:03d}",
                category=category,
                remedy_code=CATEGORY_REMEDY[category],
                title=f"{CATEGORY_TITLE[category]} (extended note {i})",
                body=(
                    f"Extended guidance for {category}: confirm the ticket category, then use "
                    f"{CATEGORY_REMEDY[category]}. Escalate if the SLA is already breached."
                ),
            )
        )

    # Fixed, seed-independent skeleton that guarantees task material exists.
    skeleton: list[dict[str, Any]] = [
        dict(customer="C-001", status="open", priority="low", category="billing",
             sla=1, text="I was charged twice for invoice INV-1001; please refund."),
        dict(customer="C-002", status="open", priority="normal", category="technical",
             sla=-2, text="The dashboard crashes with error 500 on the reports page."),
        dict(customer="C-003", status="open", priority="normal", category="shipping",
             sla=5, text="My order ORD-2003 has not arrived after two weeks."),
        dict(customer="C-001", status="open", priority="normal", category="account",
             sla=12, text="I cannot reset my password; the reset email never arrives.",
             duplicate_of=None),
        dict(customer="C-001", status="pending", priority="low", category="account",
             sla=12, text="Duplicate of T-0004: cannot reset password.",
             duplicate_of="T-0004"),
        dict(customer="C-002", status="open", priority="high", category="billing",
             sla=-4, text="Double charge on invoice INV-1006; card was billed twice."),
        dict(customer="C-004", status="resolved", priority="normal", category="refund",
             sla=0, text="Refund for order ORD-3007 already processed.",
             remedy_code=CATEGORY_REMEDY["refund"]),
        dict(customer="C-003", status="pending", priority="high", category="technical",
             sla=3, text="Export cache error blocks my weekly report."),
    ]

    tickets: dict[str, Ticket] = {}
    for i in range(n_tickets):
        tid = f"T-{i + 1:04d}"
        if i < len(skeleton):
            spec = skeleton[i]
        else:
            category = rng.choice(CATEGORIES)
            customer = rng.choice(list(customers))
            status = rng.choices(
                STATUSES, weights=(0.55, 0.15, 0.10, 0.20), k=1
            )[0]
            priority = rng.choices(PRIORITIES, weights=(0.25, 0.45, 0.20, 0.10), k=1)[0]
            sla = rng.choice([-8, -4, -1, 0, 2, 4, 8, 12, 24, 48])
            text = rng.choice(TICKET_TEMPLATES[category]).format(n=4000 + i)
            spec = dict(
                customer=customer,
                status=status,
                priority=priority,
                category=category,
                sla=sla,
                text=text,
            )
        remedy = spec.get("remedy_code")
        if spec["status"] == "resolved" and remedy is None:
            remedy = CATEGORY_REMEDY[spec["category"]]
        tickets[tid] = Ticket(
            ticket_id=tid,
            customer_id=spec["customer"],
            status=spec["status"],
            priority=spec["priority"],
            category=spec["category"],
            sla_hours_remaining=spec["sla"],
            text=spec["text"],
            remedy_code=remedy,
            duplicate_of=spec.get("duplicate_of"),
        )

    return CRMState(
        seed=seed,
        customers=customers,
        tickets=tickets,
        articles=articles,
    )
