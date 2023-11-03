from typing import List
from ..models.LLM import InferenceLLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class DistillBertLocal(InferenceLLM):

    def __init__(self, checkpoint: str):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)  
    
    def tokenize(self, sequence: str, sequences: List[str] = None) -> torch.Tensor:
        if not isinstance(sequence, str):
            raise TypeError("sequences must be a string or list of strings")
        # TODO: check for string of lists
        return self.tokenizer(sequence, return_tensors="pt")
    
    def invoke(self, tokens):
        raise not NotImplementedError()

    def batch(self):
        raise not NotImplementedError()
