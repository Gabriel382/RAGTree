import json
from pathlib import Path

from ragtree.core.config import load_config
from ragtree.preprocessing.ingest.convert_registry import CONVERTERS
import ragtree.preprocessing.ingest.converters  # noqa: F401  <-- triggers registrations

# Optional progress bar
try:
    from tqdm.auto import tqdm
except ImportError:  # fallback if tqdm is not installed
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)


# point to your project root (adapt if needed)
project_root = Path(r"/home/galencarmedeiro/git/postdoc/ragtree")


def main(
    name: str,
    truncate: bool = False,
    start: int = 0,
    distance: int | None = None,
):
    """
    Convert a dataset registered under `name` into a JSONL file.

    Parameters
    ----------
    name : str
        One of {"causalbank", "docred_causal", "eventstoryline",
                "fincausal", "maven_ere"}.

    truncate : bool, default False
        Only meaningful for 'causalbank'. If True and name == 'causalbank',
        the converter will only process a slice of lines per file:
        [start, start + distance).

    start : int, default 0
        0-based index of the first line to keep per file for 'causalbank'
        when truncate=True.

    distance : int | None, default None
        Number of lines to keep per file for 'causalbank' when truncate=True.
        If None, keeps from `start` to end of each file.
    """
    if start < 0:
        raise ValueError(f"--start must be >= 0, got {start}")
    if distance is not None and distance <= 0:
        raise ValueError(f"--distance must be > 0 when provided, got {distance}")

    cfg = load_config()
    raw_map = cfg["datasets"]["raw"]
    out_map = cfg["datasets"]["preprocessed"]

    dataset_key_map = {
        "causalbank": "CausalBank",
        "docred_causal": "DocRED",
        "eventstoryline": "EventStoryLine",
        "fincausal": "FinCausal",
        "maven_ere": "MAVEN_ERE",
    }

    if name not in dataset_key_map:
        raise KeyError(f"Unknown dataset name: {name!r}")

    raw_dir = Path(raw_map.get(dataset_key_map[name]))
    out_fp = Path(out_map[name])
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    conv_cls = CONVERTERS[name]

    # --- Instantiate converter ---
    if name == "causalbank":
        # Only CausalBank receives the truncation parameters
        conv = conv_cls(
            raw_dir,
            truncate=truncate,
            start=start,
            distance=distance,
        )
    else:
        if truncate or start != 0 or distance is not None:
            print(
                "[run_convert_dataset] Warning: --truncate/--start/--distance "
                "are only used for 'causalbank'. Ignoring them for "
                f"dataset {name!r}."
            )
        conv = conv_cls(raw_dir)

    total_written = 0

    with out_fp.open("w", encoding="utf-8") as f:
        for doc in tqdm(
            conv.iter_docs(),
            desc=f"Converting {name}",
            unit="doc",
        ):
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total_written += 1

    if name == "causalbank" and truncate:
        print(
            f"✓ Wrote {total_written} docs for {name} to {out_fp} "
            f"(truncate=True, start={start}, distance={distance})"
        )
    else:
        print(f"✓ Wrote {total_written} docs for {name} to {out_fp}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Convert a dataset to JSONL (special truncation options for CausalBank)."
    )
    ap.add_argument(
        "--name",
        required=True,
        choices=["causalbank", "docred_causal", "eventstoryline", "fincausal", "maven_ere"],
        help="Name of the dataset converter to use.",
    )
    ap.add_argument(
        "--truncate",
        action="store_true",
        help=(
            "Only for 'causalbank': if set, only convert a slice of lines per file "
            "[start, start + distance)."
        ),
    )
    ap.add_argument(
        "--start",
        type=int,
        default=0,
        help="Only for 'causalbank': 0-based index of the first line to keep per file.",
    )
    ap.add_argument(
        "--distance",
        type=int,
        default=None,
        help=(
            "Only for 'causalbank': number of lines to keep per file starting from --start. "
            "If omitted with --truncate, keeps from --start to end of each file."
        ),
    )

    args = ap.parse_args()
    main(
        name=args.name,
        truncate=args.truncate,
        start=args.start,
        distance=args.distance,
    )
