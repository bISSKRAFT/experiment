from src.quality.rouge_scorer import RougeQualityScorer
from typing import Mapping, Sequence, Tuple



def test_rouge_scorer() -> None:
    scorer = RougeQualityScorer(rouge_metrics=['rouge1', 'rouge2', 'rougeL'])
    candidates = ["This is a test", "This is another test"]
    reference = "This is a test"
    scores = scorer.compute_score(candidates, reference)
    print(scores)
    assert len(scores) == len(candidates)
    assert isinstance(scores[0], dict)
    assert isinstance(scores[0]['rouge1'].precision, float)
    assert isinstance(scores[0]['rouge1'].recall, float)
    assert isinstance(scores[0]['rouge1'].fmeasure, float)
    assert isinstance(scores[0]['rouge2'].precision, float)
    assert isinstance(scores[0]['rouge2'].recall, float)
    assert isinstance(scores[0]['rouge2'].fmeasure, float)
    assert isinstance(scores[0]['rougeL'].precision, float)
    assert isinstance(scores[0]['rougeL'].recall, float)
    assert isinstance(scores[0]['rougeL'].fmeasure, float)

def test_rouge_scorer_filter() -> None:
    scorer = RougeQualityScorer(rouge_metrics=['rouge1', 'rouge2', 'rougeL'])
    candidates = ["This is a test", "This is another test"]
    reference = "This is a test"
    scores = scorer.compute_score(candidates, reference)
    print(scores)
    filtered: Sequence= scorer.get_scores(filter_key='rouge1')
    print(filtered)
    assert len(filtered) == len(candidates)
    assert filtered[0].precision == scores[0]['rouge1'].precision

def test_rouge_scorer_filter_nokey() -> None:
    scorer = RougeQualityScorer(rouge_metrics=['rouge1', 'rouge2', 'rougeL'])
    candidates = ["This is a test", "This is another test"]
    reference = "This is a test"
    scores = scorer.compute_score(candidates, reference)
    print(scores)
    filtered = scorer.get_scores()
    print(filtered)
    assert filtered == scores
