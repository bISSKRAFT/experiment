from typing import List
from torch import device
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding
from src.models.llms.base import InferenceLLM



class Distillgpt2Local(InferenceLLM):
    """A class to represent a 🤗 DistilGPT2 language model"""

    def __init__(self, checkpoint: str):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="cuda")
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="cuda").to(device("cuda"))


    def _call(self, prompt: str) -> str:
        tokens = self._tokenize(prompt).to(device("cuda"))
        output_tokens = self.model.generate(**tokens)
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    def _tokenize(self, sequence: str | List[str]) -> BatchEncoding:
        if not isinstance(sequence, str) and not isinstance(sequence, list):
            raise TypeError("sequences must be a string or list of strings")
        # TODO: check for string of lists
        return self.tokenizer(sequence, return_tensors="pt")
    