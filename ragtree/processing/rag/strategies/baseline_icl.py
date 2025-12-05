# ragtree/processing/rag/strategies/baseline_icl.py
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

from ragtree.processing.rag.strategies.baseline_relations import (
    BaselineRelationStrategy,
)


class ICLRelationStrategy(BaselineRelationStrategy):
    """
    In-context-learning (ICL) variant of the baseline relation extractor.

    It behaves like BaselineRelationStrategy, but allows you to pass a list of
    few-shot example documents. Each example is presented to the LLM as:

      - a user message containing the same document representation as the
        baseline (title, sentence/text, entities, relation type schema)
      - an assistant message containing the *gold* relations in JSON form.

    The target document is then appended as the final user message.

    The expectation is that the LLM will imitate the examples and output JSON
    for the target document in the same schema.
    """

    def predict_relations(
        self,
        doc: Dict[str, any],
        relation_types: Optional[Sequence[str]] = None,
        *,
        few_shots: Optional[Sequence[Dict[str, any]]] = None,
    ) -> Dict[str, List[List[str]]]:
        """
        Main entrypoint.

        Parameters
        ----------
        doc : dict
            Target document.

        relation_types : optional list of str
            If provided, the schema of relation types to use.
            If None or empty, infer from doc['relations'] or fall back to
            the default causal label (same as BaselineRelationStrategy).

        few_shots : optional sequence of dict
            Example documents to use for in-context learning. They should
            already contain gold 'relations' in the normalized format.

        Returns
        -------
        Dict[str, List[List[str]]]
            Predicted relations per type, with the same schema as the baseline.
        """
        if relation_types is None or len(relation_types) == 0:
            rel_types = self._infer_relation_types_from_doc(doc)
        else:
            rel_types = list(relation_types)

        messages = self._build_messages_with_icl(
            doc,
            rel_types,
            few_shots=few_shots or [],
        )
        raw = self._call_llm(messages)
        return self._parse_llm_output(raw, rel_types)

    # ------------------------------------------------------------------
    # Message construction with ICL
    # ------------------------------------------------------------------
    def _build_messages_with_icl(
        self,
        doc: Dict[str, any],
        relation_types: Sequence[str],
        *,
        few_shots: Sequence[Dict[str, any]],
    ) -> List[Dict[str, str]]:
        """
        Build chat messages including optional few-shot examples.

        Reuses BaselineRelationStrategy._build_messages to ensure we keep the
        exact same document / entity formatting as the non-ICL baseline.

        Message pattern:

          system: <same system prompt as baseline>

          user:   <example #1 document representation>
          assistant: <gold JSON for example #1>

          user:   <example #2 document representation>
          assistant: <gold JSON for example #2>

          ...

          user:   <TARGET document representation>   <-- this is what we predict on

        If `few_shots` is empty, this reduces to the original baseline messages.
        """
        # First, build the messages for the target document using the baseline logic
        base_msgs = super()._build_messages(doc, relation_types)

        if not few_shots:
            # No ICL, just behave exactly like the baseline
            return base_msgs

        if not base_msgs or base_msgs[0].get("role") != "system":
            # Defensive: fall back to baseline behavior if message structure changes
            return base_msgs

        system_content = base_msgs[0]["content"]
        target_user_msg = base_msgs[1]  # the single user message built by baseline

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        # Build ICL examples
        for ex_doc in few_shots:
            # Use the same relation_types schema as for the target document.
            ex_msgs = super()._build_messages(ex_doc, relation_types)

            # We expect structure [system, user]; we only keep the user content.
            if len(ex_msgs) < 2 or ex_msgs[1].get("role") != "user":
                continue

            ex_user_content = ex_msgs[1]["content"]
            gold_rels = ex_doc.get("relations", {})

            messages.append({"role": "user", "content": ex_user_content})
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(gold_rels, ensure_ascii=False),
                }
            )

        # Finally, add the target document as the last user turn
        messages.append(target_user_msg)

        return messages


__all__ = [
    "ICLRelationStrategy",
]
