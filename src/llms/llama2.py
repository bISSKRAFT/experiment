from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from typing_extensions import deprecated
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BatchEncoding
from awq import AutoAWQForCausalLM
from auto_gptq import AutoGPTQForCausalLM
from src.models.llms.factories import InferenceLLMFactory
from src.llms.config.generation_config import GenerationConfigMixin
from src.models.llms.base import InferenceLLM
import torch


class Llama2Locals(Enum):
    llama2_7b_chat = "meta-llama/Llama-2-7b-chat-hf"
    llama2_13b_chat = "meta-llama/Llama-2-13b-chat-hf"
    llama2_70b_chat = "meta-llama/Llama-2-70b-chat-hf"
    llama2_7b_chat_the_bloke_awq = "TheBloke/Llama-2-7b-Chat-AWQ"
    llama2_13b_chat_the_bloke_awq = "TheBloke/Llama-2-13b-Chat-AWQ"
    llama2_70b_chat_the_bloke_awq = "TheBloke/Llama-2-70b-Chat-AWQ"
    llama_2_7b_chat_awq = "/modelcache/leos_models/meta-llama/Llama-2-7b-chat-hf-awq"
    llama_2_13b_chat_awq = "/modelcache/leos_models/meta-llama/Llama-2-13b-chat-hf-awq"
    llama_2_7b_chat_gptq = "/modelcache/leos_models/meta-llama/Llama-2-7b-chat-hf-gptq"
    llama_2_13b_chat_gptq = "/modelcache/leos_models/meta-llama/Llama-2-13b-chat-hf-gptq"



class Llama2Local(InferenceLLM):

    def __init__(self,
                 factory: Callable, 
                 checkpoint_model: str,
                 checkpoint_tokenizer: str,
                 config: Optional[AutoConfig] = None, 
                 params: Optional[Dict[str, Any]] = None):
        if not factory:
            raise ValueError("factory must be specified")
        self.model_name = checkpoint_model
        config = self._get_config(checkpoint_model, config)
        if params is None:
            params = {}
        self.checkpoint = checkpoint_model
        self.model = factory(checkpoint_model,**params, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_tokenizer, trust_remote_code=True)
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


# depracated
class Llama2LocalTheBlokeQuantized(Llama2Local):
    
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


class Llama2LocalAwq(Llama2Local):
    
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
    
class Llama2LocalGptq(Llama2Local):
    
    def _tokenize(self, sequence: List[str] | str) -> BatchEncoding:
        if not isinstance(sequence, str) and not isinstance(sequence, list):
            raise TypeError("sequences must be a string or list of strings")
        return self.tokenizer(sequence, return_tensors="pt")
    
    def _call(self, prompt: str, generation_config: Optional[GenerationConfigMixin]) -> str:
        if generation_config is None:
            generation_config = GenerationConfigMixin()
        tokens = self._tokenize(prompt).to(device="cuda")
        # GPTQ fails to set pad_token_id, when generation_config is set
        generation_config["pad_token_id"] = self.tokenizer.eos_token_id
        # GPTQ does only take **kwargs for generate!
        output_tokens = self.model.generate(**tokens, generation_config=generation_config.to_hf_generation_config())
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    
    def _get_prompt_length_in_tokens(self, prompts: List[str] | str) -> List[int]:
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self._tokenize(prompts).input_ids
        return [len(token) for token in tokens]



class Llama2LocalFactory(InferenceLLMFactory):

    @classmethod
    def create(
            cls,
            model: Llama2Locals,
            config: Optional[AutoConfig] = None
    ) -> InferenceLLM:
        # TODO: make functions each branch for more readability
        if model in [
            Llama2Locals.llama2_7b_chat,
            Llama2Locals.llama2_13b_chat,
            Llama2Locals.llama2_70b_chat
            ]:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(Llama2Local(
                AutoModelForCausalLM.from_pretrained,
                model.value,
                model.value,
                params={"device_map": "cuda:0", "torch_dtype": torch.bfloat16},
                config=config
                ))
            return cls.get_instance()
        if model in [
            Llama2Locals.llama2_7b_chat_the_bloke_awq,
            Llama2Locals.llama2_13b_chat_the_bloke_awq,
            Llama2Locals.llama2_70b_chat_the_bloke_awq
            ]:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(Llama2LocalAwq(
                AutoAWQForCausalLM.from_quantized,
                model.value,
                model.value,
                params={
                    "device_map": "auto",
                    "fuse_layers": True,
                    "trust_remote_code": False,
                    "safetensors": True,
                },
                config=config
            ))
            return cls.get_instance()
        if model in [
            Llama2Locals.llama_2_7b_chat_awq,
            Llama2Locals.llama_2_13b_chat_awq
            ]:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(Llama2LocalAwq(
                AutoAWQForCausalLM.from_pretrained,
                model.value,
                model.value,
                params={
                    "device_map": "cuda:0",
                    "trust_remote_code": True,
                    "safetensors": True,
                },
                config=config
            ))
            return cls.get_instance()
        if model in [
            Llama2Locals.llama_2_7b_chat_gptq,
            Llama2Locals.llama_2_13b_chat_gptq
            ]:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(Llama2LocalGptq(
                AutoGPTQForCausalLM.from_quantized,
                model.value,
                model.value,
                params={
                    "device": "cuda:0",
                    "trust_remote_code": True,
                    "safetensors": True,
                    "disable_exllamav2": True
                },
                config=config
            ))
            return cls.get_instance()
        raise ValueError(f"Model {model} is not supported by Llama2LocalFactory")
    
    

# class Llama2Optimum(InferenceLLM):

#     def __init__(
#             self,
#             checkpoint: str
#         ) -> None:
#         pass
        
    