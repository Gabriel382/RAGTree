# scripts/run_ontology_linking.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from ragtree.core.config import load_config  # <- adapt if your function is named differently
from ragtree.ontologies.loader import OntologyIndex
from ragtree.ontologies.mapping import OntologyEntityLinker

from tqdm import tqdm


def resolve_paths_from_config(
    config_path: Optional[Path],
    dataset_key: str,
    ontology_key: str,
    output_override: Optional[Path] = None,
) -> tuple[Path, Path, Path]:
    """
    Use default.yaml (or provided config path) to resolve:
      - input JSONL (preprocessed dataset)
      - output JSONL (either override or same as input)
      - ontology .ttl path
    """
    cfg = load_config(config_path)  # expected to return a dict-like config

    try:
        ds_pre = cfg["datasets"]["preprocessed"]
    except KeyError as e:
        raise KeyError(f"Config missing 'datasets.preprocessed' section: {e}")

    if dataset_key not in ds_pre:
        available = ", ".join(sorted(ds_pre.keys()))
        raise KeyError(
            f"Unknown dataset key '{dataset_key}'. "
            f"Available preprocessed datasets: {available}"
        )

    input_path = Path(ds_pre[dataset_key])

    try:
        ont_cfg = cfg["ontology"]
    except KeyError as e:
        raise KeyError(f"Config missing 'ontology' section: {e}")

    if ontology_key not in ont_cfg:
        available = ", ".join(sorted(ont_cfg.keys()))
        raise KeyError(
            f"Unknown ontology key '{ontology_key}'. "
            f"Available ontologies: {available}"
        )

    ontology_path = Path(ont_cfg[ontology_key])

    if output_override is not None:
        output_path = output_override
    else:
        # If no output is given, use *the same* file as input
        output_path = input_path

    return input_path, output_path, ontology_path


def run_ontology_linking(
    input_jsonl: Path,
    output_jsonl: Path,
    ontology_ttl: Path,
    backend: str = "ollama",
    top_k: int = 3,
    min_score: float = 0.3,
) -> None:
    print(f"[run_ontology_linking] Loading ontology from {ontology_ttl} ...")
    ont_index = OntologyIndex.from_turtle(ontology_ttl)

    print(f"[run_ontology_linking] Building linker with backend={backend} ...")
    linker = OntologyEntityLinker(ontology_index=ont_index, backend=backend)

    # If input and output are the same path, we’ll write to a temp file then swap
    in_place = input_jsonl.resolve() == output_jsonl.resolve()
    tmp_path: Optional[Path] = None
    if in_place:
        tmp_path = output_jsonl.with_suffix(output_jsonl.suffix + ".ontmp")
        target = tmp_path
        print(f"[run_ontology_linking] In-place mode: writing to temp {tmp_path} then replacing.")
    else:
        target = output_jsonl

    print(f"[run_ontology_linking] Reading {input_jsonl} and writing {target} ...")
    num_docs = 0

    with input_jsonl.open("r", encoding="utf-8") as fin, \
         target.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)
            links = linker.link_document(
                doc,
                top_k=top_k,
                min_score=min_score,
            )
            doc["ontology_links"] = links
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            num_docs += 1

    if in_place and tmp_path is not None:
        # replace the original file with the temp file
        tmp_path.replace(output_jsonl)
        print(f"[run_ontology_linking] Replaced {output_jsonl} with updated file.")

    print(f"[run_ontology_linking] Done. Processed {num_docs} documents.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Link entities in preprocessed JSONL to ontology concepts, using paths from default.yaml."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (e.g. config/default.yaml). "
             "If omitted, load_config() should resolve the default.",
    )

    parser.add_argument(
        "--dataset-key",
        required=True,
        help=(
            "Key under datasets.preprocessed in default.yaml "
            "(e.g. 'maven_ere', 'causalbank', ...)."
        ),
    )

    parser.add_argument(
        "--ontology-key",
        required=True,
        help=(
            "Key under ontology in default.yaml "
            "(e.g. 'framenet', 'wordnet', 'causalbank', ...)."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output JSONL file. "
            "If omitted, the script writes back to the same file as the input "
            "(datasets.preprocessed[dataset-key]), in-place."
        ),
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="ollama",
        choices=["ollama", "openrouter"],
        help="LLM backend to use for embeddings (wired via ragtree.services.llm).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of ontology concepts to keep per entity.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.3,
        help="Minimum cosine similarity to keep a candidate concept.",
    )

    args = parser.parse_args()

    input_path, output_path, ontology_path = resolve_paths_from_config(
        config_path=args.config,
        dataset_key=args.dataset_key,
        ontology_key=args.ontology_key,
        output_override=args.output,
    )

    print(f"[config] dataset-key={args.dataset_key} -> input={input_path}")
    print(f"[config] ontology-key={args.ontology_key} -> ontology={ontology_path}")
    print(f"[config] output={output_path}")

    run_ontology_linking(
        input_jsonl=input_path,
        output_jsonl=output_path,
        ontology_ttl=ontology_path,
        backend=args.backend,
        top_k=args.top_k,
        min_score=args.min_score,
    )


if __name__ == "__main__":
    main()
