# ragtree/tasks/claim_verification.py
"""Claim verification: SUPPORTS / REFUTES / NOT_ENOUGH_INFO over evidence."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseTask

__all__ = ["ClaimVerificationTask", "LABELS"]

LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")


class ClaimVerificationTask(BaseTask):
    task_type = "claim_verification"
    default_system_prompt = (
        "You verify claims against evidence. Reply with exactly two lines:\n"
        "LABEL: one of SUPPORTS, REFUTES, NOT_ENOUGH_INFO\n"
        "RATIONALE: one sentence citing the evidence references used."
    )

    def __init__(self, claim: str, **kwargs) -> None:
        super().__init__(query=claim, **kwargs)
        self.claim = claim

    def instructions(self) -> str:
        return f"Claim: {self.claim}\nDoes the evidence support or refute this claim?"

    def parse_output(self, text: str) -> dict[str, Any]:
        label = "NOT_ENOUGH_INFO"
        match = re.search(r"\b(SUPPORTS|REFUTES|NOT_ENOUGH_INFO)\b", text or "")
        if match:
            label = match.group(1)
        rationale = ""
        rat_match = re.search(r"RATIONALE:\s*(.+)", text or "", re.DOTALL)
        if rat_match:
            rationale = rat_match.group(1).strip()
        return {"label": label, "rationale": rationale}
