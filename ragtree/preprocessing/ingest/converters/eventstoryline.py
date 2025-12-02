from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Set

from ragtree.preprocessing.ingest.convert_registry import BaseConverter, register



@register("eventstoryline")
class EventStoryLineConverter(BaseConverter):
    """
    Converter for the EventStoryLine / ECB+ evaluation format (v1.0).

    It reads:
      - event-level relations from:
            evaluation_format/<split>_corpus/v1.0/event_mentions_extended
      - document text / tokens from:
            ECB+_LREC2014/ECB+/<topic>/<doc_stem>.xml  (ecbplus version if
            available, otherwise ecb)

    Output JSONL format is intentionally aligned with the MAVEN_ERE
    normalisation used in this project, with the following keys:

        document_id : str
        title       : str
        text        : str
        type        : str  ("train" / "test" / "dev" depending on split)
        sentences   : List[str]
        tokens      : List[List[str]]
        entities    : Dict[event_id, {
                         "type": Optional[str],
                         "mentions": [{
                             "id": str,
                             "trigger_word": str,
                             "sent_id": int,
                             "offset": [int, int],
                         }]
                      }]
        relations   : Dict[relation_type, List[[event_id, event_id]]]
    """

    def __init__(self, raw_dir: Path) -> None:
        super().__init__(raw_dir)
        # Root for the ECB+ tokenised documents
        self._ecb_root = self.raw_dir / "ECB+_LREC2014" / "ECB+"
        # Root for the evaluation-format annotations
        self._eval_root = self.raw_dir / "evaluation_format"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _hash_event_id(doc_key: str, span_str: str) -> str:
        """
        Create a deterministic EVENT_ identifier from a document key
        and a span string such as "64" or "183_184".
        """
        h = hashlib.md5(f"{doc_key}:{span_str}".encode("utf-8")).hexdigest()
        return f"EVENT_{h}"

    @staticmethod
    def _hash_mention_id(doc_key: str, span_str: str, idx: int = 0) -> str:
        h = hashlib.md5(f"{doc_key}:{span_str}:{idx}".encode("utf-8")).hexdigest()
        return h

    @staticmethod
    def _parse_span(span_str: str) -> List[int]:
        """
        Turn "64" or "183_184" into a list of integer token ids.
        """
        parts = span_str.split("_")
        return [int(p) for p in parts if p]

    @staticmethod
    def _corpus_type_from_dirname(dirname: str) -> str:
        """
        Map directory name like "test_corpus" or "training_corpus" to
        a simple split label.
        """
        prefix = dirname.split("_", 1)[0].lower()
        if prefix in {"train", "training"}:
            return "train"
        if prefix in {"test"}:
            return "test"
        if prefix in {"dev", "development"}:
            return "dev"
        return prefix

    def _parse_tokens(
        self, doc_xml: Path
    ) -> Tuple[Dict[int, Dict[str, Any]], List[str], List[List[str]]]:
        """
        Parse ECB+/ECB+plus XML and return:
          - tokens_by_id: t_id -> { "text", "sent_id", "idx" }
          - sentences:    list of sentence strings
          - sent_tokens:  list of list-of-token-strings per sentence
        """
        tree = ET.parse(doc_xml)
        root = tree.getroot()

        tmp: List[Tuple[int, int, int, str]] = []
        for tok in root.iter("token"):
            t_id = int(tok.attrib["t_id"])
            sent_id = int(tok.attrib["sentence"])
            num = int(tok.attrib["number"])
            text = tok.text or ""
            tmp.append((sent_id, num, t_id, text))

        # Sort by sentence then position
        tmp.sort(key=lambda x: (x[0], x[1]))

        tokens_by_id: Dict[int, Dict[str, Any]] = {}
        sent_tokens_dict: Dict[int, List[str]] = defaultdict(list)

        for sent_id, num, t_id, text in tmp:
            idx = len(sent_tokens_dict[sent_id])
            sent_tokens_dict[sent_id].append(text)
            tokens_by_id[t_id] = {
                "text": text,
                "sent_id": sent_id,
                "idx": idx,
            }

        # Normalise sentences and tokens to lists sorted by sentence id
        if sent_tokens_dict:
            max_sent = max(sent_tokens_dict.keys())
        else:
            max_sent = -1

        sentences: List[str] = []
        sent_tokens: List[List[str]] = []
        for s in range(max_sent + 1):
            toks = sent_tokens_dict.get(s, [])
            sent_tokens.append(toks)
            sentences.append(" ".join(toks))

        return tokens_by_id, sentences, sent_tokens

    def _parse_event_relations(
        self, rel_file: Path
    ) -> Tuple[Set[str], Dict[str, List[Tuple[str, str]]]]:
        """
        Parse an event_mentions_extended file.

        Each non-empty line is assumed to have the structure:
            <src_span> <tgt_span> <RELATION_TYPE>

        e.g.:
            64 94 FALLING_ACTION
            183_184 196 PRECONDITION
        """
        spans: Set[str] = set()
        rel_by_type: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        with rel_file.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    # Unexpected format; skip it gracefully.
                    continue
                src_span, tgt_span = parts[0], parts[1]
                rel_type = parts[-1]

                spans.add(src_span)
                spans.add(tgt_span)
                rel_by_type[rel_type].append((src_span, tgt_span))

        return spans, rel_by_type

    def _build_entities(
        self,
        doc_key: str,
        spans: Iterable[str],
        tokens_by_id: Dict[int, Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Create the entities dict and a mapping span_str -> event_id.

        Entities follow the MAVEN_ERE-like structure:

            {
              "EVENT_xxx": {
                  "type": None,
                  "mentions": [{
                      "id": "...",
                      "trigger_word": "token(s)",
                      "sent_id": int,
                      "offset": [start, end],
                  }]
              },
              ...
            }
        """
        entities: Dict[str, Any] = {}
        span2event: Dict[str, str] = {}

        for span_str in sorted(spans):
            token_ids = self._parse_span(span_str)
            if not token_ids:
                continue

            # Ensure all tokens exist and (ideally) share the same sentence
            meta = [tokens_by_id.get(tid) for tid in token_ids]
            if any(m is None for m in meta):
                # Span refers to missing token ids – skip
                continue

            sent_ids = {m["sent_id"] for m in meta if m is not None}
            if not sent_ids:
                continue
            # For robustness, if a span crosses sentences we just pick the first.
            sent_id = min(sent_ids)

            positions = [m["idx"] for m in meta if m is not None]
            start = min(positions)
            end = max(positions) + 1  # exclusive

            # Use tokens ordered by their position inside the sentence
            ordered = sorted(meta, key=lambda m: m["idx"])
            trigger_word = " ".join(m["text"] for m in ordered)

            event_id = self._hash_event_id(doc_key, span_str)
            mention_id = self._hash_mention_id(doc_key, span_str, 0)

            entities[event_id] = {
                "type": None,  # EventStoryLine does not provide a fine-grained event type
                "mentions": [
                    {
                        "id": mention_id,
                        "trigger_word": trigger_word,
                        "sent_id": sent_id,
                        "offset": [start, end],
                    }
                ],
            }
            span2event[span_str] = event_id

        return entities, span2event

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def iter_docs(self) -> Iterable[Dict[str, Any]]:
        """
        Walk through all available v1.0 event_mentions_extended files and
        yield one JSON-like dict per document, aligned with the MAVEN_ERE
        normalisation format.
        """
        if not self._eval_root.exists():
            return

        # Loop over *corpus directories, e.g. "test_corpus", "training_corpus"
        for corpus_dir in sorted(self._eval_root.glob("*_corpus")):
            if not corpus_dir.is_dir():
                continue

            split_label = self._corpus_type_from_dirname(corpus_dir.name)

            v1_dir = corpus_dir / "v1.0"
            if not v1_dir.exists():
                continue

            ev_ext_root = v1_dir / "event_mentions_extended"
            if not ev_ext_root.exists():
                continue

            # Topic-level subdirectories: 1, 12, 13, ...
            for topic_dir in sorted(ev_ext_root.iterdir()):
                if not topic_dir.is_dir():
                    continue
                topic_id = topic_dir.name  # e.g. "1"

                # Each file is something like "1_1ecbplus.xml"
                for rel_file in sorted(topic_dir.glob("*ecbplus.xml")):
                    doc_stem = rel_file.stem  # "1_1ecbplus"
                    doc_key = doc_stem

                    # Corresponding ECB+ document with tokens
                    # First try the "ecbplus" file, then fall back to "ecb".
                    ecb_doc = self._ecb_root / topic_id / f"{doc_stem}.xml"
                    if not ecb_doc.exists():
                        # 1_1ecbplus -> 1_1ecb
                        alt_stem = doc_stem.replace("ecbplus", "ecb")
                        ecb_doc = self._ecb_root / topic_id / f"{alt_stem}.xml"
                        if not ecb_doc.exists():
                            # If we really cannot find the text, skip this document.
                            continue

                    tokens_by_id, sentences, sent_tokens = self._parse_tokens(ecb_doc)

                    # Collect spans + relation definitions
                    spans, rel_by_type_span = self._parse_event_relations(rel_file)

                    # Build entities dict and span -> EVENT_ mapping
                    entities, span2event = self._build_entities(
                        doc_key, spans, tokens_by_id
                    )

                    # Map relations from spans to event ids
                    relations: Dict[str, List[List[str]]] = defaultdict(list)
                    for rel_type, pairs in rel_by_type_span.items():
                        for src_span, tgt_span in pairs:
                            src_e = span2event.get(src_span)
                            tgt_e = span2event.get(tgt_span)
                            if src_e is None or tgt_e is None:
                                continue
                            relations[rel_type].append([src_e, tgt_e])

                    # Build a simple document text as concatenation of sentences
                    full_text = " ".join(sentences)

                    yield {
                        "document_id": f"EventStoryLine - {doc_key}",
                        "title": doc_key,
                        "text": full_text,
                        "type": split_label,
                        "sentences": sentences,
                        "tokens": sent_tokens,
                        "entities": entities,
                        "relations": dict(relations),
                    }
