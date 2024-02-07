import gc
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BatchEncoding
from awq import AutoAWQForCausalLM
from src.models.llms.factories import InferenceLLMFactory
from src.llms.config.generation_config import GenerationConfigMixin
from src.models.llms.base import InferenceLLM
import torch


class Llama2Locals(Enum):
    llama2_7b_chat = "meta-llama/Llama-2-7b-chat-hf"
    llama2_13b_chat = "meta-llama/Llama-2-13b-chat-hf"
    llama2_70b_chat = "meta-llama/Llama-2-70b-chat-hf"
    llama2_7b_chat_awq_quantized = "TheBloke/Llama-2-7b-Chat-AWQ"


class Llama2Local(InferenceLLM):

    __INSTANCE = None

    def __init__(self,
                 factory: Callable, 
                 checkpoint: str,
                 config: Optional[AutoConfig] = None, 
                 params: Optional[Dict[str, Any]] = None):
        if not factory:
            raise ValueError("factory must be specified")
        self.model_name = checkpoint
        config = self._get_config(checkpoint, config)
        if params is None:
            params = {}
        self.model = factory(checkpoint,**params, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.config = self.model.config.to_dict()

    def _get_config(self, checkpoint: str, config: Optional[AutoConfig] = None) -> AutoConfig:
        if config is None:
            return AutoConfig.from_pretrained(checkpoint)
        return config
    
    def _get_model(self, checkpoint: str, config: AutoConfig, compiling: bool = False):
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, 
            config=config,
            device_map="cuda:0")
        if compiling:
            model = torch.compile(model)
        return model

    def _call(self, prompt: str, generation_config: Optional[GenerationConfigMixin]) -> str:
        if generation_config is None:
            generation_config = GenerationConfigMixin()
        tokens = self._tokenize(prompt)
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


class Llama2LocalQuantized(Llama2Local):
    
    def _tokenize(self, sequence: List[str] | str) -> BatchEncoding:
        if not isinstance(sequence, str) and not isinstance(sequence, list):
            raise TypeError("sequences must be a string or list of strings")
        return self.tokenizer(sequence, return_tensors="pt").input_ids.to(device="cuda")
    
    def _call(self, prompt: str, generation_config: Optional[GenerationConfigMixin]) -> str:
        if generation_config is None:
            generation_config = GenerationConfigMixin()
        tokens = self._tokenize(prompt).to(device="cuda")
        output_tokens = self.model.generate(tokens, generation_config=generation_config.to_hf_generation_config())
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]


class Llama2LocalFactory(InferenceLLMFactory):

    __INSTANCE = None

    @classmethod
    def create(
            cls,
            model: Llama2Locals,
            config: Optional[AutoConfig] = None
    ) -> InferenceLLM:
        if model in [
            Llama2Locals.llama2_7b_chat,
            Llama2Locals.llama2_13b_chat,
            Llama2Locals.llama2_70b_chat
            ]:
            instance_check = cls.__check_for_instance(model)
            if instance_check:
                return instance_check
            cls.__flush()
            cls.__INSTANCE = Llama2Local(
                AutoModelForCausalLM.from_pretrained,
                Llama2Locals.llama2_7b_chat.value,
                params={"device_map": "cuda:0"},
                config=config
                )
            return cls.__INSTANCE
        if model == Llama2Locals.llama2_7b_chat_awq_quantized:
            instance_check = cls.__check_for_instance(model)
            if instance_check:
                return instance_check
            cls.__flush()
            cls.__INSTANCE = Llama2LocalQuantized(
                AutoAWQForCausalLM.from_quantized,
                Llama2Locals.llama2_7b_chat_awq_quantized.value,
                params={
                    "device_map": "auto",
                    "fuse_layers": True,
                    "trust_remote_code": False,
                    "safetensors": True,
                },
                config=config
            )
            return cls.__INSTANCE
        raise ValueError(f"Model {model} is not supported by Llama2LocalFactory")

    @classmethod   
    def __check_for_instance(cls, model: Llama2Locals) -> InferenceLLM | None:
        if cls.__INSTANCE is not None and cls.__INSTANCE.model_name == model.value:
            return cls.__INSTANCE
        return None
    
    @classmethod
    def __flush(cls):
        print("Llama2LocalFactory: FLUSHING INSTANCE...")
        del cls.__INSTANCE
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    

# class Llama2Optimum(InferenceLLM):

#     def __init__(
#             self,
#             checkpoint: str
#         ) -> None:
#         pass
        
    