# ragtree/preprocessing/ingest/converters/fincausal.py
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Dict, Any, List, Tuple
import csv

from ragtree.preprocessing.ingest.convert_registry import BaseConverter, register


@register("fincausal")
class FinCausalConverter(BaseConverter):
    """
    Converter for all FinCausal CSV files under raw_dir.

    Output schema is EXACTLY the same as MavenEREConverter:

      {
        "document_id": str,
        "title": str,
        "text": str,
        "type": str,            # here: original CSV file name (e.g. 'test_predictions.csv')
        "sentences": [str],
        "tokens": [[str]],
        "entities": {
          "<EVENT_ID>": {
            "type": "CAUSE" | "EFFECT",
            "mentions": [
              {
                "id": str,
                "trigger_word": str,
                "sent_id": int,
                "offset": [start, end] | None   # token indices in tokens[sent_id]
              },
              ...
            ]
          },
          ...
        },
        "relations": {
          "CAUSE": [[<EVENT_ID_CAUSE>, <EVENT_ID_EFFECT>], ...]
        }
      }

    - Entities are phrases (cause/effect).
    - Only 'CAUSE' relation type is used.
    - One JSONL line per row in any FinCausal CSV.
    """

    # ------------------------------------------------------------------
    # Main iterator
    # ------------------------------------------------------------------
    def iter_docs(self) -> Iterator[Dict[str, Any]]:
        files = list(Path(self.raw_dir).glob("**/*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found under {self.raw_dir}")

        for fp in files:
            source_type = fp.name  # e.g. "train.csv", "test_predictions.csv"
            for row in self._iter_csv_rows(fp):
                doc = self._convert_row(row=row, source_type=source_type)
                if doc is not None:
                    yield doc

    # ------------------------------------------------------------------
    # CSV reading with delimiter detection
    # ------------------------------------------------------------------
    def _iter_csv_rows(self, fp: Path):
        """
        Yield dictionaries for each row in a CSV file,
        detecting delimiter among ';' and ','.
        """
        with fp.open("r", encoding="utf-8") as f:
            sample = f.read(4096)
            f.seek(0)

            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=";,")
                reader = csv.DictReader(f, dialect=dialect)
            except csv.Error:
                # Fallback: try ';', then ','
                f.seek(0)
                reader = csv.DictReader(f, delimiter=";")
                if not reader.fieldnames or len(reader.fieldnames) == 1:
                    f.seek(0)
                    reader = csv.DictReader(f, delimiter=",")

            for row in reader:
                # Skip completely empty rows
                if not any(
                    isinstance(v, str) and v.strip()
                    for v in row.values()
                ):
                    continue
                yield row

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_col(row: Dict[str, str], *candidates: str) -> str | None:
        """
        Case-insensitive column access: try several candidate names.
        """
        lower_map = {k.lower(): k for k in row.keys()}
        for cand in candidates:
            key = lower_map.get(cand.lower())
            if key is not None:
                val = row.get(key)
                if val is not None:
                    return val
        return None

    def _make_event_id(self, role: str, clean_text: str, index_id: str) -> str:
        """
        Build an event id like EVENT_<hash>, based on:
          - role: "CAUSE" or "EFFECT"
          - index_id: row index/id if available
          - clean_text: full sentence text

        This keeps IDs unique and still fully compatible with MAVEN_ERE style.
        """
        base = f"{role}|{index_id}|{clean_text}"
        h = self._text_hash(base, length=16)   # from BaseConverter
        return f"EVENT_{h}"

    def _convert_row(
        self,
        row: Dict[str, str],
        source_type: str,
    ) -> Dict[str, Any] | None:
        """
        Convert a single FinCausal row into the common schema.

        If we don't have any usable text, returns None and the row is skipped.
        Otherwise a document is created even if entities/relations are empty.
        """
        # Try Sentence / sentence / Text / text
        sentence_raw = (
            self._get_col(row, "Sentence", "sentence", "Text", "text") or ""
        ).strip()
        if not sentence_raw:
            return None

        clean_text, spans = self._strip_tags_and_get_spans(sentence_raw)

        sentences: List[str] = [clean_text]
        tokens: List[List[str]] = [clean_text.split()]

        # Cause / Effect from columns (if available)
        cause_phrase = (
            self._get_col(row, "Cause", "cause", "CAUSE") or ""
        ).strip() or None
        effect_phrase = (
            self._get_col(row, "Effect", "effect", "EFFECT") or ""
        ).strip() or None

        # Fallback from <e1>/<e2> spans if columns are empty
        if not cause_phrase and "e1" in spans:
            s, e = spans["e1"]
            cause_phrase = clean_text[s:e].strip() or None

        if not effect_phrase and "e2" in spans:
            s, e = spans["e2"]
            effect_phrase = clean_text[s:e].strip() or None

        entities: Dict[str, Any] = {}
        relations: Dict[str, List[List[str]]] = {"CAUSE": []}

        # Helper to map phrase -> token offsets in tokens[0]
        def find_offset(phrase: str | None) -> List[int] | None:
            if not phrase:
                return None
            phrase_tokens = phrase.split()
            if not phrase_tokens:
                return None

            sent_tokens = tokens[0]
            n = len(sent_tokens)
            m = len(phrase_tokens)

            for i in range(n - m + 1):
                if sent_tokens[i : i + m] == phrase_tokens:
                    return [i, i + m]
            return None  # allowed to be None

        index_id = (
            self._get_col(row, "Index", "index", "ID", "id", "id_sentence") or ""
        ).strip()

        eid_map: Dict[str, str] = {}

        # Cause entity
        if cause_phrase:
            eid_cause = self._make_event_id("CAUSE", clean_text, index_id)
            offset = find_offset(cause_phrase)
            entities[eid_cause] = {
                "type": "CAUSE",
                "mentions": [
                    {
                        "id": f"{index_id}_e1" if index_id else "e1",
                        "trigger_word": cause_phrase,
                        "sent_id": 0,
                        "offset": offset,  # may be None
                    }
                ],
            }
            eid_map["cause"] = eid_cause

        # Effect entity
        if effect_phrase:
            eid_effect = self._make_event_id("EFFECT", clean_text, index_id)
            offset = find_offset(effect_phrase)
            entities[eid_effect] = {
                "type": "EFFECT",
                "mentions": [
                    {
                        "id": f"{index_id}_e2" if index_id else "e2",
                        "trigger_word": effect_phrase,
                        "sent_id": 0,
                        "offset": offset,  # may be None
                    }
                ],
            }
            eid_map["effect"] = eid_effect

        # If we have both sides, add a CAUSE relation
        if "cause" in eid_map and "effect" in eid_map:
            relations["CAUSE"].append([eid_map["cause"], eid_map["effect"]])

        document_id = self.make_document_id("FinCausal", clean_text)
        title = index_id or document_id

        return {
            "document_id": document_id,
            "title": title,
            "text": clean_text,
            "type": source_type,       # e.g. "test_predictions.csv"
            "sentences": sentences,
            "tokens": tokens,
            "entities": entities,
            "relations": relations,
        }

    # ------------------------------------------------------------------
    # Strip <e1>/<e2> tags and compute spans in cleaned text
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_tags_and_get_spans(s: str) -> Tuple[str, Dict[str, Tuple[int, int]]]:
        """
        Remove <e1>, </e1>, <e2>, </e2> tags from the string
        and return:
          - cleaned string
          - spans dict: {"e1": (start_char, end_char), "e2": (...)}
            indices are in the CLEANED text.
        """
        out_chars: List[str] = []
        spans: Dict[str, Tuple[int, int]] = {}

        i = 0
        out_pos = 0
        n = len(s)

        while i < n:
            if s.startswith("<e1>", i):
                i += 4
                start = out_pos
                while i < n and not s.startswith("</e1>", i):
                    out_chars.append(s[i])
                    i += 1
                    out_pos += 1
                end = out_pos
                spans["e1"] = (start, end)
                if i < n and s.startswith("</e1>", i):
                    i += 5
            elif s.startswith("<e2>", i):
                i += 4
                start = out_pos
                while i < n and not s.startswith("</e2>", i):
                    out_chars.append(s[i])
                    i += 1
                    out_pos += 1
                end = out_pos
                spans["e2"] = (start, end)
                if i < n and s.startswith("</e2>", i):
                    i += 5
            elif s.startswith("</e1>", i):
                i += 5
            elif s.startswith("</e2>", i):
                i += 5
            else:
                out_chars.append(s[i])
                i += 1
                out_pos += 1

        cleaned = "".join(out_chars)
        return cleaned, spans
