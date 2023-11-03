import torch
import pytest
from ...LLMs.testLLM import DistillBertLocal

@pytest.mark.parametrize("sequence, expected_result", [
    ("Hello, my dog is cute", True),
    ])

def test_tokenize(sequence, expected_result):
    model = DistillBertLocal("distilbert-base-uncased-finetuned-sst-2-english")
    tokens =  model.tokenize(sequence)
    assert (tokens is not None) == expected_result

def test_invoke():
    pass