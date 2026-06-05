# scripts/run_triple_kg_rag_relations.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from ragtree.core.config import load_config
from ragtree.processing.orchestrators.relations_runner import (
    PreparedContext,
    RunnerLLMSections,
    run_relation_experiment,
)
from ragtree.processing.kg_rag.kg_loader import load_local_graphstore
from ragtree.processing.kg_rag.triple_kg_retriever import TripleKGRetriever, TripleKGRetrieverParams
from ragtree.processing.rag.strategies.triple_kg_rag_relations import TripleKGRagRelationStrategy


def _parse_doc_types(arg: str) -> Sequence[str] | str:
    if not arg or arg == "all":
        return "all"
    items = [x.strip() for x in arg.split(",")]
    items = [x for x in items if x]
    return items or "all"


def _autopick_kg_file(cfg: Dict[str, Any], dataset_key: str) -> Path:
    kg_root = Path(cfg["paths"].get("kg", "data/kg"))
    cands = sorted(kg_root.glob(f"{dataset_key}__*__kg.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(
            f"No KG artifact found for '{dataset_key}' in {kg_root}. "
            f"Expected pattern: {dataset_key}__types=...__kg.json"
        )
    return cands[0]


def _collect_few_shots(
    input_path: Path,
    *,
    shot_type: str,
    shot_num: int,
    shot_skip: int = 0,
    shot_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if shot_num <= 0:
        return []

    few_shots: List[Dict[str, Any]] = []
    seen_after_type = 0
    considered = 0

    with input_path.open("r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc=f"[triple_kg_rag] Collecting few-shots (type={shot_type})", unit="doc"):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)

            if doc.get("type") != shot_type:
                continue

            seen_after_type += 1
            if shot_skip and seen_after_type <= shot_skip:
                continue

            if shot_limit is not None and considered >= shot_limit:
                break
            considered += 1

            rels = doc.get("relations")
            if not isinstance(rels, dict) or not rels:
                continue

            few_shots.append(doc)
            if len(few_shots) >= shot_num:
                break

    print(
        f"[triple_kg_rag] few-shots: requested={shot_num} collected={len(few_shots)} "
        f"type={shot_type} shot_skip={shot_skip} shot_limit={shot_limit}"
    )
    return few_shots


def prepare_triple_kg_rag_context(
    input_path: Path,
    cfg: Dict[str, Any],
    *,
    kg_path: Path,
    max_hops: int,
    max_triples: int,
    include_in_edges: bool,
    max_sentences_in_prompt: Optional[int],
    max_triples_in_text: int,
    shot_type: str,
    shot_num: int,
    shot_skip: int,
    shot_limit: Optional[int],
) -> PreparedContext:
    gs = load_local_graphstore(kg_path)

    retriever = TripleKGRetriever(
        gs,
        params=TripleKGRetrieverParams(
            max_hops=max_hops,
            max_triples=max_triples,
            include_in_edges=include_in_edges,
        ),
    )

    strategy_kwargs = {
        "retriever": retriever,
        "max_sentences_in_prompt": max_sentences_in_prompt,
        "max_triples_in_text": max_triples_in_text,
    }

    few_shots = _collect_few_shots(
        input_path,
        shot_type=shot_type,
        shot_num=shot_num,
        shot_skip=shot_skip,
        shot_limit=shot_limit,
    )

    predict_kwargs = {"few_shots": few_shots}
    return PreparedContext(strategy_kwargs=strategy_kwargs, predict_kwargs=predict_kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description="Run triple_kg_rag relation extraction (reuse built LocalGraphStore KG).")

    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset-key", required=True)
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--model", type=str, default=None)

    p.add_argument("--output-format", choices=["full", "pred-only"], default="full")
    p.add_argument("--doc-type-filter", default="all")
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)

    # KG
    p.add_argument("--kg-path", type=Path, default=None)
    p.add_argument("--kg-max-hops", type=int, default=1)
    p.add_argument("--kg-max-triples", type=int, default=80)
    p.add_argument("--kg-include-in-edges", action="store_true", default=True)
    p.add_argument("--no-kg-include-in-edges", action="store_false", dest="kg_include_in_edges")

    # prompt size
    p.add_argument("--max-sentences-in-prompt", type=int, default=None)
    p.add_argument("--max-triples-in-text", type=int, default=80)

    # few-shot
    p.add_argument("--shot-type", type=str, default="train")
    p.add_argument("--shot-num", type=int, default=0)
    p.add_argument("--shot-skip", type=int, default=0)
    p.add_argument("--shot-limit", type=int, default=None)

    args = p.parse_args()
    cfg = load_config(args.config)

    kg_file = args.kg_path if args.kg_path else _autopick_kg_file(cfg, args.dataset_key)

    sections = RunnerLLMSections(
        llm_section="triple_kg_rag",
        prompt_section="triple_kg_rag",
        system_prompt_key="triple_kg_rag_docre",
    )

    def _prep(input_path: Path, cfg2: Dict[str, Any]) -> PreparedContext:
        return prepare_triple_kg_rag_context(
            input_path,
            cfg2,
            kg_path=kg_file,
            max_hops=args.kg_max_hops,
            max_triples=args.kg_max_triples,
            include_in_edges=bool(args.kg_include_in_edges),
            max_sentences_in_prompt=args.max_sentences_in_prompt,
            max_triples_in_text=args.max_triples_in_text,
            shot_type=args.shot_type,
            shot_num=args.shot_num,
            shot_skip=args.shot_skip,
            shot_limit=args.shot_limit,
        )

    run_relation_experiment(
        strategy_cls=TripleKGRagRelationStrategy,
        config_path=args.config,
        dataset_key=args.dataset_key,
        backend=args.backend,
        model=args.model,
        cli_relation_types=None,
        output_format=args.output_format,
        doc_type_filter=_parse_doc_types(args.doc_type_filter),
        skip=args.skip,
        limit=args.limit,
        sections=sections,
        prepare_context_fn=_prep,
    )


if __name__ == "__main__":
    main()