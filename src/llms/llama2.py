from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.llms.base import InferenceLLM
import torch

class Llama2Local(InferenceLLM):
    def __init__(self, checkpoint: str):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, torch_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(torch.device("cuda"))

    def _call(self, prompt: str) -> str:
        tokens = self._tokenize(prompt).to(torch.device("cuda"))
        output_tokens = self.model.generate(**tokens)
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    def _tokenize(self, sequence: List[str] | str):
        if not isinstance(sequence, str) and not isinstance(sequence, list):
            raise TypeError("sequences must be a string or list of strings")
        return self.tokenizer(sequence, return_tensors="pt")
    