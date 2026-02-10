# ragtree/preprocessing/ingest/converters/maven_ere.py
from pathlib import Path
from typing import Iterator, Dict, Any
import json

from ragtree.preprocessing.ingest.convert_registry import BaseConverter, register


@register("maven_ere")
class MavenEREConverter(BaseConverter):
    """
    Converter for MAVEN-ERE JSONL files.

    Input:  *.jsonl with fields like:
      - id, title, tokens (list[list[str]]), sentences (list[str]),
        events (list[{"id", "type", "mention": [...] }]),
        causal_relations: { "CAUSE": [...], "PRECONDITION": [...], ... }

    Output (per line in preprocessed JSONL):
      {
        "document_id": "MAVEN_ERE - <hash>",
        "title":       <title or document_id>,
        "text":        <full text>,
        "type":        "train" | "validation" | "test" | "unknown",
        "sentences":   [...],
        "tokens":      [[...], ...],

        // One entity per EVENT_* (event-centric), with all mentions
        "entities": {
          "EVENT_xxx": {
            "type": "Control" | "Self_motion" | ...,
            "mentions": [
              {
                "id": "<mention_id>",
                "trigger_word": "<surface trigger from JSON>",
                "sent_id": <int>,
                "offset": [start, end]  // token indices in tokens[sent_id]
              },
              ...
            ]
          },
          ...
        },

        // Relations only from causal_relations, using EVENT_* ids directly
        "relations": {
          "CAUSE":        [["EVENT_a", "EVENT_b"], ...],
          "PRECONDITION": [["EVENT_c", "EVENT_d"], ...],
          ...
        }
      }
    """

    # If True, drop any document that ends up with no causal relations at all
    drop_docs_without_causal: bool = True

    def iter_docs(self) -> Iterator[Dict[str, Any]]:
        # MAVEN-ERE files are JSONL
        files = list(Path(self.raw_dir).glob("**/*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No JSONL found under {self.raw_dir}")

        for fp in files:
            # Infer split (type) from filename
            stem = fp.stem.lower()
            if "train" in stem:
                split = "train"
            elif "dev" in stem or "valid" in stem or "val" in stem:
                split = "validation"
            elif "test" in stem:
                split = "test"
            else:
                split = "unknown"

            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)

                    # Sentences and tokens
                    sentences = obj.get("sentences")
                    tokens = obj.get("tokens")

                    # Build text from sentences if possible, else from tokens
                    if sentences:
                        text = " ".join(sentences)
                    elif tokens:
                        text = " ".join(" ".join(sent) for sent in tokens)
                    else:
                        text = obj.get("title", "")

                    # Fallbacks for sentences/tokens if missing
                    if sentences is None and tokens:
                        sentences = [" ".join(sent) for sent in tokens]
                    if tokens is None and sentences:
                        tokens = [s.split() for s in sentences]

                    # document_id and title
                    document_id = self.make_document_id("MAVEN_ERE", text)
                    title = obj.get("title") or document_id

                    # ------- Entities: event-centric, with all mentions -------

                    events = obj.get("events", []) or []
                    entities: Dict[str, Any] = {}

                    for ev in events:
                        ev_id = ev.get("id")
                        if not ev_id:
                            # skip malformed events without id
                            continue

                        ev_type = ev.get("type")
                        mentions_raw = ev.get("mention", []) or []

                        mentions = []
                        for m in mentions_raw:
                            mentions.append({
                                "id": m.get("id"),
                                "trigger_word": m.get("trigger_word"),
                                "sent_id": m.get("sent_id"),
                                "offset": m.get("offset"),
                            })

                        entities[ev_id] = {
                            "type": ev_type,
                            "mentions": mentions,
                        }

                    # ------- Relations: only from causal_relations, using EVENT_* ids -------

                    relations: Dict[str, list[list[str]]] = {}
                    causal_rels = obj.get("causal_relations", {}) or {}
                    valid_ids = set(entities.keys())

                    for rel_type, pairs in causal_rels.items():
                        rel_list = relations.setdefault(rel_type, [])
                        for head_id, tail_id in pairs:
                            # only keep relations where both endpoints are known events
                            if head_id in valid_ids and tail_id in valid_ids:
                                rel_list.append([head_id, tail_id])

                    # Optionally drop docs with no causal relations at all
                    if self.drop_docs_without_causal:
                        # "no causal relations" means: relations dict is empty
                        # OR all lists inside are empty
                        if not relations or not any(rel_list for rel_list in relations.values()):
                            continue

                    yield {
                        "document_id": document_id,
                        "title": title,
                        "text": text,
                        "type": split,
                        "sentences": sentences or [],
                        "tokens": tokens or [],
                        "entities": entities,
                        "relations": relations,
                    }
