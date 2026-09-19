"""Synthetic policy corpus used by the RAG layer."""

POLICIES = [
    {
        "policy_id": "P-001",
        "title": "Device Protection - Accidental Damage",
        "text": "Accidental physical damage, including cracked screens and accidental drops, is eligible for coverage when the device is enrolled and the incident occurs during the active protection period. Cosmetic damage that does not affect device functionality is excluded. Deductibles may apply depending on device tier.",
    },
    {
        "policy_id": "P-002",
        "title": "Device Protection - Theft and Loss",
        "text": "Theft may be covered when the device is reported promptly and required documentation is supplied. Loss without evidence of qualifying circumstances may be excluded. Replacement limits and deductibles depend on the customer's plan.",
    },
    {
        "policy_id": "P-003",
        "title": "Claims Review and Fraud Screening",
        "text": "Claims can be routed to additional review when transaction patterns, claim frequency, claim value, or other data signals are inconsistent with normal activity. An anomaly or review signal does not by itself establish fraud. Claims remain subject to policy terms and human review procedures.",
    },
    {
        "policy_id": "P-004",
        "title": "Battery and Wear Coverage",
        "text": "Normal battery degradation and ordinary wear are generally not treated as accidental damage. Coverage for battery-related service depends on the specific protection plan and eligibility rules in effect at the time of the claim.",
    },
]
