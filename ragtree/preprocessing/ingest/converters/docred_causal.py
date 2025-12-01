from __future__ import annotations

from pathlib import Path
from typing import Iterator, Dict, Any
import json
import hashlib

from ragtree.preprocessing.ingest.convert_registry import BaseConverter, register


@register("docred_causal")
class DocREDConverter(BaseConverter):
    """
    Converter for DocRED (dev.json, test.json, train_annotated.json, train_distant.json)
    into the SAME schema used by the MAVEN-ERE converter.

    Input (per DocRED doc):
      {
        "sents":      list[list[str]],
        "vertexSet":  list[list[ { "name", "pos", "sent_id", "type" } ]],
        "labels":     [ { "h", "t", "r", "evidence": [...] }, ... ]   # except in test.json
        "title":      str
      }

    Output (per line in preprocessed JSONL), Maven-ERE-like:
      {
        "document_id": "DocRED - <hash>",
        "title":       <title or document_id>,
        "text":        <full text>,
        "type":        "dev" | "test" | "train_annotated" | "train_distant",
        "sentences":   [...],
        "tokens":      [[...], ...],

        "entities": {
          "Event_xxx": {
            "type": "PER" | "LOC" | "ORG" | ... (DocRED NER type),
            "mentions": [
              {
                "id": "<mention_id>",
                "trigger_word": "<surface form>",
                "sent_id": <int>,
                "offset": [start, end]  // token indices in tokens[sent_id]
              },
              ...
            ]
          },
          ...
        },

        "relations": {
          "P17 : country":        [["Event_a", "Event_b"], ...],
          "P159 : headquarters location": [["Event_c", "Event_d"], ...],
          ...
        }
      }

    Notes:
    - rel_info.json is loaded from self.raw_dir and used to map PXX -> name.
    - test.json has no labels, so "relations" will be {} for those docs.
    - We KEEP docs even if they have 0 relations (unlike MAVEN-ERE's drop_docs_without_causal).
    """

    # For DocRED we *do not* drop docs without relations (test has none)
    drop_docs_without_causal: bool = False

    def _load_rel_info(self) -> Dict[str, str]:
        """
        Load rel_info.json from the same directory as the DocRED splits.
        default.yaml should point raw.DocRED to something like:
          data/raw/DocRED/DocRED
        and rel_info.json must live there.
        """
        rel_path = Path(self.raw_dir) / "rel_info.json"
        if not rel_path.exists():
            raise FileNotFoundError(f"rel_info.json not found under {self.raw_dir}")
        with rel_path.open("r", encoding="utf-8") as f:
            rel_info = json.load(f)
        # Ensure keys and values are strings
        return {str(k): str(v) for k, v in rel_info.items()}

    @staticmethod
    def _make_event_id(document_id: str, split: str, ent_idx: int, surface: str) -> str:
        """
        Deterministic 'Event_<hash>' id based on doc context + entity index + surface form.
        """
        raw = f"{document_id}||{split}||{ent_idx}||{surface}"
        h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        return f"Event_{h}"

    def iter_docs(self) -> Iterator[Dict[str, Any]]:
        raw_dir = Path(self.raw_dir)

        # Load relation info from rel_info.json (PXX -> name)
        rel_info = self._load_rel_info()

        # (split_name, filename) pairs
        split_files = [
            ("dev", "dev.json"),
            ("test", "test.json"),
            ("train_annotated", "train_annotated.json"),
            ("train_distant", "train_distant.json"),
        ]

        for split, fname in split_files:
            fpath = raw_dir / fname
            if not fpath.exists():
                # Soft warning, skip missing splits
                print(f"[DocREDConverter] Warning: {fpath} not found, skipping split '{split}'.")
                continue

            with fpath.open("r", encoding="utf-8") as f:
                data = json.load(f)

            for obj in data:
                # -------- Tokens & sentences --------
                tokens = obj.get("sents") or []  # list[list[str]]
                sentences = [" ".join(sent) for sent in tokens]
                text = " ".join(sentences)

                # -------- document_id & title --------
                document_id = self.make_document_id("DocRED", text)
                title = obj.get("title") or document_id

                # -------- Entities: from vertexSet --------
                vertex_set = obj.get("vertexSet") or []
                entities: Dict[str, Any] = {}
                ent_index_to_id: Dict[int, str] = {}

                for ent_idx, cluster in enumerate(vertex_set):
                    if not cluster:
                        continue

                    # DocRED uses same entity type for all mentions in cluster -> take first
                    ev_type = cluster[0].get("type", "UNK")

                    mentions = []
                    for m_idx, m in enumerate(cluster):
                        sent_id = m["sent_id"]
                        start, end = m["pos"]
                        # surface form from tokens
                        trigger_word = " ".join(tokens[sent_id][start:end])
                        mention_id = f"{ent_idx}_m{m_idx}"

                        mentions.append(
                            {
                                "id": mention_id,
                                "trigger_word": trigger_word,
                                "sent_id": sent_id,
                                "offset": [start, end],
                            }
                        )

                    # Use the first mention surface for hashing
                    surface = mentions[0]["trigger_word"] if mentions else f"ENT_{ent_idx}"
                    ent_id = self._make_event_id(document_id, split, ent_idx, surface)

                    entities[ent_id] = {
                        "type": ev_type,   # PER / LOC / ORG / ...
                        "mentions": mentions,
                    }
                    ent_index_to_id[ent_idx] = ent_id

                # -------- Relations: from labels (if present) --------
                relations: Dict[str, list[list[str]]] = {}
                labels = obj.get("labels") or []

                for lab in labels:
                    prop_id = lab["r"]  # e.g. "P17"
                    h_idx = lab["h"]    # head entity index (int)
                    t_idx = lab["t"]    # tail entity index (int)

                    head_id = ent_index_to_id.get(h_idx)
                    tail_id = ent_index_to_id.get(t_idx)
                    if head_id is None or tail_id is None:
                        # skip malformed indices
                        continue

                    # Build "PXX : name" label
                    name = rel_info.get(prop_id)
                    if name:
                        rel_type = f"{prop_id} : {name}"
                    else:
                        rel_type = prop_id  # fallback if not found

                    rel_list = relations.setdefault(rel_type, [])
                    rel_list.append([head_id, tail_id])

                # Optionally, you could drop docs with no relations,
                # but for DocRED we KEEP them (especially test split).
                if self.drop_docs_without_causal:
                    if not relations or not any(rel_list for rel_list in relations.values()):
                        continue

                yield {
                    "document_id": document_id,
                    "title": title,
                    "text": text,
                    "type": split,          # "dev" | "test" | "train_annotated" | "train_distant"
                    "sentences": sentences,
                    "tokens": tokens,
                    "entities": entities,
                    "relations": relations,
                }
