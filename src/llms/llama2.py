from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BatchEncoding
from src.llms.config.generation_config import GenerationConfigMixin
from src.models.llms.base import InferenceLLM
import torch

class Llama2Local(InferenceLLM):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Llama2Local, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, 
                 checkpoint: str,
                 config: Optional[AutoConfig] = None, 
                 compiling: bool = False):
        if self.__initialized: return
        self.model_name = checkpoint
        config = self._get_config(checkpoint, config)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = self._get_model(checkpoint, config, compiling)
        self.config = self.model.config.to_dict()
        self.__initialized = True

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

    def _call(self, prompt: str, generation_config: Optional[GenerationConfigMixin]) -> str:
        if generation_config is None:
            generation_config = GenerationConfigMixin()
        tokens = self._tokenize(prompt).to(device="cuda:0")
        output_tokens = self.model.generate(**tokens, generation_config=generation_config.to_hf_generation_config())
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    def _tokenize(self, sequence: List[str] | str) -> BatchEncoding:
        if not isinstance(sequence, str) and not isinstance(sequence, list):
            raise TypeError("sequences must be a string or list of strings")
        return self.tokenizer(sequence, return_tensors="pt")
    
    def _get_prompt_length_in_tokens(self, prompts: List[str] | str) -> List[int]:
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self._tokenize(prompts)["input_ids"]
        return [len(token) for token in tokens]


# class Llama2Optimum(InferenceLLM):

#     def __init__(
#             self,
#             checkpoint: str
#         ) -> None:
#         pass
        
    