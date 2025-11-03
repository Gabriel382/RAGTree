# ragtree/postprocessing/eval.py
from pathlib import Path
import json
from ragtree.evaluation.ner_eval import ner_metrics
from ragtree.evaluation.re_eval import re_metrics
from ragtree.evaluation.alignment_eval import alignment_metrics

def evaluate(gold_path: str | Path, pred_path: str | Path, ontology=None) -> dict:
    gold = [json.loads(l) for l in Path(gold_path).open("r", encoding="utf-8")]
    pred = [json.loads(l) for l in Path(pred_path).open("r", encoding="utf-8")]

    ner_res = ner_metrics(gold, pred)
    re_res = re_metrics(gold, pred)
    align_res = alignment_metrics(pred, ontology)

    return {
        "ner": ner_res,
        "re": re_res,
        "alignment": align_res,
    }
