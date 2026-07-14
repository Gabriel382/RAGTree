# ragtree/processing/rag/strategies/chain_of_thought.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Optional

from ragtree.processing.rag.strategies.baseline_relations import BaselineRelationStrategy

class ChainOfThoughtRelationStrategy(BaselineRelationStrategy):
    """
    Two-step Chain-of-Thought (CoT) strategy:
    1. Ask the LLM to think step-by-step (reasoning only, no JSON)
    2. Feed the reasoning back and ask for JSON only

    Optional:
      print_cot=True -> prints reasoning per document (for debugging)
    """

    # ------------------------------------------------------------
    # PUBLIC PREDICTION METHOD
    # ------------------------------------------------------------
    def predict_relations(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
        *,
        print_cot: bool = False,
    ) -> Dict[str, List[List[str]]]:

        # ----- CALL 1: produce reasoning -----
        messages_reason = self._build_messages_reasoning(doc, relation_types)
        reasoning_output = self._call_llm(messages_reason)

        if print_cot:
            print("\n====== CoT Reasoning ======")
            print(reasoning_output)
            print("===========================\n")

        # ----- CALL 2: convert to JSON -----
        messages_json = self._build_messages_json(reasoning_output, relation_types)
        raw_json = self._call_llm(messages_json)

        # Parse JSON like baseline
        return self._parse_llm_output(raw_json, relation_types)

    # ------------------------------------------------------------
    # MESSAGE GENERATORS
    # ------------------------------------------------------------

    def _build_messages_reasoning(
        self,
        doc: Dict[str, Any],
        relation_types: Sequence[str],
    ) -> List[Dict[str, str]]:
        """
        Step 1: Build messages for reasoning.
        Reuse the baseline structure but add a reasoning instruction.
        """
        title = doc.get("title", "")
        text = doc.get("text", "")
        entities = doc.get("entities", {})

        system_prompt = (
            self.llm_config.system_prompt
            + "\n\nYou MUST NOT output JSON yet. "
              "Explain step-by-step how entities might relate according to the candidate relation types."
        )

        user_message = (
            f"Document Title: {title}\n"
            f"Text: {text}\n\n"
            f"Entities:\n{json.dumps(entities, ensure_ascii=False, indent=2)}\n\n"
            f"Candidate relation types: {relation_types}\n\n"
            "Think step-by-step and explain your reasoning. "
            "Do NOT output JSON yet."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]

    def _build_messages_json(
        self,
        reasoning_output: str,
        relation_types: Sequence[str],
    ) -> List[Dict[str, str]]:
        """
        Step 2: Ask the model to output ONLY the JSON.
        Provide the reasoning back as context.
        """

        user_message = (
            "Here is your earlier reasoning:\n\n"
            f"{reasoning_output}\n\n"
            "Now output ONLY the JSON object representing the predicted relations. "
            "Do not include any explanation. "
            "Use only the relation types: "
            f"{relation_types}.\n"
        )

        return [
            {"role": "system", "content": self.llm_config.system_prompt},
            {"role": "user",   "content": user_message},
        ]


__all__ = ["ChainOfThoughtRelationStrategy"]
