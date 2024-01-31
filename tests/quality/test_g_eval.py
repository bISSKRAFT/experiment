from src.quality.g_eval import GEvalQualityScorer


def test_compute_score() -> None:
    scorer = GEvalQualityScorer(model_name="gpt-3.5-turbo-1106")
    source = "This is a test and the contents of this string should be summarized and the summary should be evaluated."
    scores = scorer.compute_score(
        candidates="This is a test",
        source=source
    )
    assert isinstance(scores, dict)
    print(scores)
    dimensions = {'relevance', 'fluency', 'coherence', 'consistency'}
    assert scores.keys() == dimensions

def test_calculate_score() -> None:
    scorer = GEvalQualityScorer(model_name="gpt-3.5-turbo-1106")
    respones = ['2', '1', '2', '2', '1 ', '3', '3', '1', '3', '3', 'Coherence: 3', '2', '2', '1', '1', '55']
    print(respones)
    score = scorer._calucate_score(respones, len(respones))
    assert isinstance(score, float)
    assert score >= 0.0
    print(score)

def test_make_request() -> None:
    scorer = GEvalQualityScorer(model_name="gpt-3.5-turbo-1106")
    prompt = "This is a test and the contents of this string should be summarized and the summary should be evaluated."
    res, count = scorer._make_reqeuest(prompt)
    print(res)
    assert isinstance(res, list)
    assert isinstance(count, int)
    assert count == 20

def test_check_for_digits() -> None:
    scorer = GEvalQualityScorer(model_name="gpt-3.5-turbo-1106")
    respones = ['2', '1', '2', '2', '1 ', '3', '3', '1', '3', '3', 'Coherence: 3', '2', '\n\n- Coherence:', '1', 'Sent for revision', '1', '5']
    res, count = scorer._check_for_digits(respones)
    print("cleaned: ",res)
    assert isinstance(res, list)
    assert isinstance(count, int)
    assert count == 15
    print(res)