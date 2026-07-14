"""End-to-end relation extraction over every tiny benchmark fixture.

Golden-metric harness: a gold-echo provider must reach F1 = 1.0 and an
empty provider F1 = 0.0 on each dataset slice, proving the task prompt,
JSON parsing, normalization and the historical relation metrics agree.
"""

import json
from pathlib import Path

import pytest

from ragtree.core.pipeline import RAGTreePipeline
from ragtree.evaluation.relation_evaluator import RelationEvaluator
from ragtree.integrations.llms import MockLLMProvider
from ragtree.tasks import RelationExtractionTask, results_from_strategy

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "relations"
DATASETS = ["causalbank", "docred_causal", "eventstoryline", "fincausal"]


def load_docs(name: str) -> list[dict]:
    path = FIXTURE_DIR / f"{name}_tiny.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("dataset", DATASETS)
def test_gold_echo_provider_scores_perfect_f1(dataset):
    for doc in load_docs(dataset):
        relation_types = list(doc["relations"].keys())
        task = RelationExtractionTask(relation_types, document=doc)
        pipeline = RAGTreePipeline(
            generator=MockLLMProvider(reply=json.dumps(doc["relations"])),
            evaluator=RelationEvaluator(),
        )
        result = pipeline.run(task, reference=doc["relations"])
        assert result.metrics["f1"] == pytest.approx(1.0), (dataset, doc["document_id"])
        assert result.metrics["precision"] == pytest.approx(1.0)
        assert result.metrics["recall"] == pytest.approx(1.0)


@pytest.mark.parametrize("dataset", DATASETS)
def test_empty_provider_scores_zero_f1(dataset):
    doc = load_docs(dataset)[0]
    relation_types = list(doc["relations"].keys())
    task = RelationExtractionTask(relation_types, document=doc)
    pipeline = RAGTreePipeline(
        generator=MockLLMProvider(reply="{}"), evaluator=RelationEvaluator()
    )
    result = pipeline.run(task, reference=doc["relations"])
    assert result.metrics["f1"] == 0.0
    assert result.metrics["recall"] == 0.0


@pytest.mark.parametrize("dataset", DATASETS)
def test_prompt_carries_document_and_entities(dataset):
    doc = load_docs(dataset)[0]
    task = RelationExtractionTask(list(doc["relations"].keys()), document=doc)
    provider = MockLLMProvider(reply="{}")
    RAGTreePipeline(generator=provider).run(task)
    prompt = provider.calls[0][1]["content"]
    assert doc["document_id"] in prompt
    first_entity = next(iter(doc.get("entities") or {}), None)
    if first_entity:
        assert first_entity in prompt


def test_wrapped_benchmark_strategy_preserves_pred_relations_format(monkeypatch):
    """Run an existing benchmark strategy unchanged through the task layer."""
    from ragtree.processing.rag.base_strategy import BaseRelationStrategy, LLMBackendConfig
    from ragtree.processing.rag.strategies.baseline_relations import BaselineRelationStrategy

    docs = load_docs("fincausal")[:2]

    class GoldByTextStub:
        def chat(self, messages, **kwargs):
            prompt = " ".join(m.get("content", "") for m in messages)
            for doc in docs:
                probe = str(doc.get("text", ""))[:60]
                if probe and probe in prompt:
                    return json.dumps(doc["relations"])
            return "{}"

    monkeypatch.setattr(
        BaseRelationStrategy, "_init_llm_client", lambda self: GoldByTextStub()
    )
    strategy = BaselineRelationStrategy(
        LLMBackendConfig(backend="stub", model="stub-model")
    )

    relation_types = list(docs[0]["relations"].keys())
    results = results_from_strategy(strategy, docs, relation_types=relation_types)

    assert len(results) == len(docs)
    evaluator = RelationEvaluator()
    for doc, result in zip(docs, results):
        prediction = result.artifacts["prediction"]
        assert prediction["document_id"] == doc["document_id"]
        assert set(prediction["relations"].keys()) == set(relation_types)
        report = evaluator.evaluate(result, reference=doc["relations"])
        assert report.metrics["f1"] == pytest.approx(1.0), doc["document_id"]
