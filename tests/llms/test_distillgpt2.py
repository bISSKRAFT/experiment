from ...src.llms.testLLM import Distillgpt2Local

# @pytest.mark.parametrize("sequence, expected_result", [
#    ("Hello, my dog is cute", True),
#    ])

def test_invoke():
    model = Distillgpt2Local("distilgpt2")
    result = model.invoke("Hello, my dog is cute")
    assert isinstance(result, str)
