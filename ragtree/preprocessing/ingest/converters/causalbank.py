from __future__ import annotations

from pathlib import Path
from typing import Iterator, Dict, Any, Set, Tuple
import json
import re

from ragtree.preprocessing.ingest.convert_registry import BaseConverter, register


@register("causalbank")
class CausalBankConverter(BaseConverter):
    """
    Converter for CausalBank text files into the unified MAVEN-ERE-like JSONL format.

    Expected folder structure under raw_dir:

      raw_dir/
        Lexical_Cause_Effect_Graph.txt
        because_mode/
          <connector_name_1>
          <connector_name_2>
          ...
        therefore_mode/
          <connector_name_1>
          <connector_name_2>
          ...

    Example line in a because_mode file (tab-separated):
        caused by    A0002233.exe error is principally    memory issue, driver issue, ...

      - col[0] = connector (e.g. "caused by")
      - col[1] = phrase 1  (we'll call it p1)
      - col[2] = phrase 2  (we'll call it p2)

    We build a single sentence:
        text = f"{p1} {connector} {p2}"

    Lexical_Cause_Effect_Graph.txt example line:
        favorite->finally    1351    0.08    0.09

      - left_lemma  = "favorite"
      - right_lemma = "finally"
      - we ignore numeric/stat columns for now

    Output per document:

      {
        "document_id": "CausalBank - <hash>",
        "title":       "<connector> - <p1> | <p2>",
        "text":        "<p1> <connector> <p2>",
        "type":        "<source_filename>",  # e.g. "because_of", "caused_by", ...
        "sentences":   ["..."],
        "tokens":      [[...]],

        "entities": {
          "EVENT_<hash>": {
            "type": "entity_type",
            "mentions": [
              {
                "id": "<doc_local_mention_id>",
                "trigger_word": "<surface_token>",
                "sent_id": 0,
                "offset": [start, end]  # token indices in tokens[0]
              },
              ...
            ]
          },
          ...
        },

        "relations": {
          "BECAUSE":   [["EVENT_x", "EVENT_y"], ...],
          "THEREFORE": [["EVENT_a", "EVENT_b"], ...]
        }
      }

    Notes:
      - We use a lightweight normalization/lemmatization applied BOTH to the
        lexical graph lemmas and to the document tokens, so that e.g.
        "current" in the graph will match "currently" in the text.
      - Only single-token lemmas from the lexical graph are used for now
        (multiword entries are ignored to keep matching tractable).
    """

    drop_docs_without_causal: bool = True

    def __init__(self, raw_dir: str | Path):
        super().__init__(raw_dir)
        self.lex_graph_path = self.raw_dir / "Lexical_Cause_Effect_Graph.txt"

        if not self.lex_graph_path.is_file():
            raise FileNotFoundError(
                f"Lexical_Cause_Effect_Graph.txt not found under {self.raw_dir}"
            )

        # Build:
        #  - a set of normalized lemmas that define what is considered an entity
        #  - a set of (lemma1, lemma2) pairs that define possible causal links
        (
            self.entity_lemmas,
            self.causal_pairs,
        ) = self._load_lexical_cause_effect_graph(self.lex_graph_path)

    # -------------------------------------------------------------------------
    # Normalization / Lemmatization
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_token(token: str) -> str:
        """
        Lightweight normalization used as a "lemma-like" form:

          - lowercase
          - strip leading/trailing non-alphanumerics
          - strip common English suffixes: -ing, -ed, -ly, -es, -s
            (only if token is long enough, to avoid over-stemming)

        This is intentionally simple and dependency-free. You can replace
        this with a proper lemmatizer later if desired.
        """
        t = token.lower()
        # strip punctuation at boundaries
        t = re.sub(r"^[^a-z0-9]+", "", t)
        t = re.sub(r"[^a-z0-9]+$", "", t)

        # simple suffix stripping
        for suf in ("ing", "ed", "ly", "es", "s"):
            if t.endswith(suf) and len(t) > len(suf) + 2:
                t = t[: -len(suf)]
                break

        return t

    # -------------------------------------------------------------------------
    # Loading Lexical_Cause_Effect_Graph.txt
    # -------------------------------------------------------------------------

    def _load_lexical_cause_effect_graph(
        self, path: Path
    ) -> tuple[Set[str], Set[Tuple[str, str]]]:
        """
        Parse Lexical_Cause_Effect_Graph.txt and build:

          - entity_lemmas: set of normalized lemmas
          - causal_pairs: set of (lemma1, lemma2) pairs, both normalized

        Only entries where both sides are single-token (no spaces) are kept,
        to keep matching feasible without a heavy phrase-matcher.
        """
        entity_lemmas: Set[str] = set()
        causal_pairs: Set[Tuple[str, str]] = set()

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # expected: "lemma1->lemma2\tcount\tp1\tp2..."
                # we only care about the "lemma1->lemma2" part
                first_field = line.split("\t", 1)[0]
                if "->" not in first_field:
                    continue

                left_raw, right_raw = first_field.split("->", 1)

                # ignore multi-word entries for now (to avoid huge phrase matching)
                if " " in left_raw.strip() or " " in right_raw.strip():
                    continue

                left_norm = self._normalize_token(left_raw)
                right_norm = self._normalize_token(right_raw)

                if not left_norm or not right_norm:
                    continue

                entity_lemmas.add(left_norm)
                entity_lemmas.add(right_norm)
                causal_pairs.add((left_norm, right_norm))

        return entity_lemmas, causal_pairs

    # -------------------------------------------------------------------------
    # Main iteration
    # -------------------------------------------------------------------------

    def iter_docs(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over all lines in because_mode/ and therefore_mode/ and yield
        normalized documents.
        """
        because_dir = self.raw_dir / "because_mode"
        therefore_dir = self.raw_dir / "therefore_mode"

        if not because_dir.is_dir() and not therefore_dir.is_dir():
            raise FileNotFoundError(
                f"Neither because_mode/ nor therefore_mode/ found under {self.raw_dir}"
            )

        # Process BECAUSE_MODE files
        if because_dir.is_dir():
            for doc in self._iter_mode_docs(because_dir, rel_type="BECAUSE"):
                yield doc

        # Process THEREFORE_MODE files
        if therefore_dir.is_dir():
            for doc in self._iter_mode_docs(therefore_dir, rel_type="THEREFORE"):
                yield doc

    # -------------------------------------------------------------------------

    def _iter_mode_docs(
        self, mode_dir: Path, rel_type: str
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over all files in a mode directory (because_mode or therefore_mode)
        and build documents.

        Each line is assumed to be tab-separated:

          connector <TAB> phrase1 <TAB> phrase2 [<TAB> ...]

        We construct:
          text = f"{phrase1} {connector} {phrase2}"
        """
        for fp in mode_dir.iterdir():
            if not fp.is_file():
                continue

            # connector name can be inferred from filename if needed
            connector_name = fp.name.replace("_", " ")
            doc_type = fp.name  # <-- use the file name as the 'type'

            with fp.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    cols = line.split("\t")
                    if len(cols) < 3:
                        # malformed line, skip
                        continue

                    connector = cols[0].strip() or connector_name
                    phrase1 = cols[1].strip()
                    phrase2 = cols[2].strip()

                    # Build a single sentence with connector in between
                    sentence = f"{phrase1} {connector} {phrase2}"
                    sentences = [sentence]
                    tokens = [sentence.split()]

                    text = sentence
                    document_id = self.make_document_id("CausalBank", text)
                    title = f"{connector} - {phrase1} | {phrase2}"

                    # ---------------- Entities via lemma matching ----------------

                    entities: Dict[str, Any] = {}
                    # lemma -> list of (token_index, surface_form)
                    lemma_mentions: Dict[str, list[tuple[int, str]]] = {}
                    # lemma -> entity_id (EVENT_<hash>)
                    lemma_to_entid: Dict[str, str] = {}

                    # We have a single sentence with index 0
                    for idx, tok in enumerate(tokens[0]):
                        norm = self._normalize_token(tok)
                        if not norm:
                            continue
                        if norm not in self.entity_lemmas:
                            continue
                        lemma_mentions.setdefault(norm, []).append((idx, tok))

                    # Build entity structures with hashed IDs
                    mention_counter = 0
                    for lemma, occurrences in lemma_mentions.items():
                        # Unique hash per (lemma, document text)
                        hash_input = f"{lemma}|{text}"
                        ent_hash = self._text_hash(hash_input)
                        ent_id = f"EVENT_{ent_hash}"
                        lemma_to_entid[lemma] = ent_id

                        ent_mentions = []
                        for pos, surface in occurrences:
                            mention_id = f"m{mention_counter}"
                            mention_counter += 1
                            ent_mentions.append(
                                {
                                    "id": mention_id,
                                    "trigger_word": surface,
                                    "sent_id": 0,
                                    "offset": [pos, pos + 1],
                                }
                            )

                        entities[ent_id] = {
                            "type": "entity_type",
                            "mentions": ent_mentions,
                        }

                    # ---------------- Relations via lexical pairs ----------------

                    relations: Dict[str, list[list[str]]] = {}

                    if entities and lemma_to_entid:
                        edges_set: Set[Tuple[str, str]] = set()

                        lemmas_in_doc = list(lemma_to_entid.keys())

                        for i, l1 in enumerate(lemmas_in_doc):
                            for j, l2 in enumerate(lemmas_in_doc):
                                if i == j:
                                    continue
                                if (l1, l2) in self.causal_pairs:
                                    head = lemma_to_entid[l1]
                                    tail = lemma_to_entid[l2]
                                    edges_set.add((head, tail))

                        if edges_set:
                            rel_list = []
                            for head, tail in sorted(edges_set):
                                rel_list.append([head, tail])
                            relations[rel_type] = rel_list

                    # Optionally drop docs without any relations
                    if self.drop_docs_without_causal:
                        if not relations or not any(rel_list for rel_list in relations.values()):
                            continue

                    yield {
                        "document_id": document_id,
                        "title": title,
                        "text": text,
                        "type": doc_type,          # <-- file name as type
                        "sentences": sentences,
                        "tokens": tokens,
                        "entities": entities,
                        "relations": relations,
                    }
