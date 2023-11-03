import torch
from typing import List
import pytest
from ...LLMs.testLLM import Distillgpt2Local

@pytest.mark.parametrize("sequence, expected_result", [
    ("Hello, my dog is cute", True),
    ])

def test_tokenize(sequence, expected_result):
    model = Distillgpt2Local("distilgpt2")
    tokens =  model.tokenize(sequence)
    assert (tokens is not None) == expected_result

def test_tokenize_exception():
    model = Distillgpt2Local("distilgpt2")
    with pytest.raises(TypeError):
        model.tokenize(1)

def test_invoke():
    model = Distillgpt2Local("distilgpt2")
    tokens = model.tokenize("Hello, my dog is cute")
    result = model.invoke(tokens)
    print("result: ", model.tokenizer.batch_decode(result, skip_special_tokens=True))
    assert result is not None
    assert isinstance(model.tokenizer.batch_decode(result, skip_special_tokens=True)[0], str)