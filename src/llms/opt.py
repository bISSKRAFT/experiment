from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BatchEncoding
from typing import Any, Callable, Dict, List, Optional
import torch
from awq import AutoAWQForCausalLM
from auto_gptq import AutoGPTQForCausalLM


from src.models.llms.factories import InferenceLLMFactory
from src.llms.config.generation_config import GenerationConfigMixin
from src.models.llms.base import InferenceLLM

class OptLocals(Enum):
    opt_13b = "facebook/opt-13b"
    opt_temp = ""


class OptLocal(InferenceLLM):

    def __init__(self,
                 factory: Callable, 
                 checkpoint_model: str,
                 checkpoint_tokenizer: str,
                 config: Optional[AutoConfig] = None, 
                 params: Optional[Dict[str, Any]] = None,
                 model_name: Optional[str] = None):
        if not factory:
            raise ValueError("factory must be specified")
        if model_name is None:
            model_name = checkpoint_model
        self.model_name = model_name   
        config = self._get_config(checkpoint_model, config)
        if params is None:
            params = {}
        self.checkpoint = checkpoint_model
        print("used parameters:" ,params)
        self.model = factory(checkpoint_model,**params, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_tokenizer, trust_remote_code=True)
        self.config = self.model.config.to_dict()

    def _get_config(self, checkpoint: str, config: Optional[AutoConfig] = None) -> AutoConfig:
        if config is None:
            return AutoConfig.from_pretrained(checkpoint)
        return config
    
    def _call(self, prompt: str, generation_config: Optional[GenerationConfigMixin]) -> str:
        if generation_config is None:
            generation_config = GenerationConfigMixin()
        tokens = self._tokenize(prompt)
        print("my config: ", generation_config.generation_config)
        print("hf config: ", generation_config.to_hf_generation_config().to_dict())
        output_tokens = self.model.generate(tokens, generation_config=generation_config.to_hf_generation_config())
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    def _tokenize(self, sequence: List[str] | str) -> BatchEncoding:
        if not isinstance(sequence, str) and not isinstance(sequence, list):
            raise TypeError("sequences must be a string or list of strings")
        return self.tokenizer(sequence, return_tensors="pt").input_ids.to(device="cuda:0")
    
    def _get_prompt_length_in_tokens(self, prompts: List[str] | str) -> List[int]:
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self._tokenize(prompts)
        return [len(token) for token in tokens]
    

class OptLocalFactory(InferenceLLMFactory):

    @classmethod
    def create(
        cls,
        model: OptLocals,
        config: Optional[AutoConfig] = None,
    ) -> InferenceLLM:
        if model == OptLocals.opt_13b:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(OptLocal(
                factory=AutoModelForCausalLM.from_pretrained,
                checkpoint_model=model.value,
                checkpoint_tokenizer=model.value,
                config=config,
                params={
                    "device_map": "cuda:0",
                    "torch_dtype": torch.float16,
                    },
                model_name=model.value
            ))
            return cls.get_instance()
        raise ValueError(f"Model {model} not supported by {cls.__name__}")