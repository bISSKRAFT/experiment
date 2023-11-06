from typing import List
from ..models.llms.base import InferenceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Distillgpt2Local(InferenceLLM):

    def __init__(self, checkpoint: str):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)

    def _call(self, prompt: str) -> str:
        tokens = self._tokenize(prompt)
        output_tokens = self.model.generate(**tokens)
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    
    def _tokenize(self, sequence: str, sequences: List[str] = None) -> torch.Tensor:
        if not isinstance(sequence, str):
            raise TypeError("sequences must be a string or list of strings")
        # TODO: check for string of lists
        return self.tokenizer(sequence, return_tensors="pt")
    
