from src.utils.kpis.performance import calculate_tokens_per_second



def test_calulate_tokens_per_second():
    assert calculate_tokens_per_second([100, 200], 10, [1, 2], 0) == [90, 95]