import json
from ragtree.core.config import load_config
from ragtree.preprocessing.ingest.convert_registry import CONVERTERS
import ragtree.preprocessing.ingest.converters  # noqa: F401  <-- triggers registrations
from pathlib import Path
import sys, os

# point to your project root
project_root = Path(r"C:\Users\henri\Documents\git\post-doc\ragtree")

def main(name: str):
    cfg = load_config()
    raw_map = cfg["datasets"]["raw"]
    out_map = cfg["datasets"]["preprocessed"]
    raw_dir = Path(raw_map.get({"causalbank":"CausalBank",
                                "docred_causal":"DocRED",
                                "eventstoryline":"EventStoryLine",
                                "fincausal":"FinCausal",
                                "maven_ere":"MAVEN_ERE"}[name]))
    out_fp = Path(out_map[name])
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    conv_cls = CONVERTERS[name]
    conv = conv_cls(raw_dir)

    with out_fp.open("w", encoding="utf-8") as f:
        for doc in conv.iter_docs():
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"✓ Wrote {name} to {out_fp}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    choices=["causalbank","docred_causal","eventstoryline","fincausal","maven_ere"])
    args = ap.parse_args()
    main(args.name)
