from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BatchEncoding
from src.models.llms.base import InferenceLLM
import torch

class Llama2Local(InferenceLLM):

    def __init__(self, 
                 checkpoint: str,
                 config: Optional[AutoConfig] = None, 
                 compiling: bool = False):
        self.model_name = checkpoint
        config = self._get_config(checkpoint, config)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = self._get_model(checkpoint, config, compiling)
        self.config = self.model.config.to_dict()

    def _get_config(self, checkpoint: str, config: Optional[AutoConfig] = None) -> AutoConfig:
        if config is None:
            return AutoConfig.from_pretrained(checkpoint)
        return config
    
    def _get_model(self, checkpoint: str, config: AutoConfig, compiling: bool = False):
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, 
            config=config).to(device="cuda:0")
        if compiling:
            model = torch.compile(model)
        return model

    def _call(self, prompt: str) -> str:
        tokens = self._tokenize(prompt).to(device="cuda:0")
        output_tokens = self.model.generate(**tokens, max_length=50)
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    def _tokenize(self, sequence: List[str] | str) -> BatchEncoding:
        if not isinstance(sequence, str) and not isinstance(sequence, list):
            raise TypeError("sequences must be a string or list of strings")
        return self.tokenizer(sequence, return_tensors="pt")
    
    def _get_prompt_length_in_tokens(self, prompts: List[str]) -> List[int]:
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self._tokenize(prompts)["input_ids"]
        return [len(token) for token in tokens]
    
@NotImplemented
class Llama2Optimum(InferenceLLM):

    def __init__(
            self,
            checkpoint: str
        ) -> None:
        pass
        
    