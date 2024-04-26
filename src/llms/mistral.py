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

class MistralLocals(Enum):
    mistral_7b_instruct = "mistralai/Mistral-7B-Instruct-v0.2"
    mistral_7b_instruct_gptq = "/modelcache/leos_models/mistral/mistralai/Mistral-7B-Instruct-v0.2-gptq"
    mistral_7b_instruct_awq = "/modelcache/leos_models/mistral/mistralai/Mistral-7B-Instruct-v0.2-awq"
    mistral_7b_instruct_bnb = "/modelcache/leos_models/mistral/mistralai/Mistral-7B-Instruct-v0.2-bnb"
    mistral_7b_instruct_flash_attn = "mistralai/Mistral-7B-Instruct-v0.2-flash_attn"


class MistralLocal(InferenceLLM):
    
    def __init__(self,
                 factory: Callable,
                 checkpoint_model: str,
                 checkpoint_tokenizer: str,
                 config: Optional[AutoConfig] = None,
                 params: Optional[Dict[str, Any]] = None,
                 model_name: Optional[str] = None,
                 ):
        if not factory:
            raise ValueError("factory must be provided")
        if model_name is None:
            model_name = checkpoint_model
        self.model_name = model_name
        config = self._get_config(checkpoint_model, config)
        if params is None:
            params = {}
        print("used params: ", params)
        self.checkpoint = checkpoint_model
        self.model = factory(checkpoint_model,**params, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_tokenizer)
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
    
    def _tokenize(self, sequence: List[str] | str) -> BatchEncoding:
        if not isinstance(sequence, str) and not isinstance(sequence, list):
            raise TypeError("sequences must be a string or list of strings")
        return self.tokenizer(sequence, return_tensors="pt").input_ids.to(device="cuda:0")
    
    def _get_prompt_length_in_tokens(self, prompts: List[str] | str) -> List[int]:
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self._tokenize(prompts)
        return [len(token) for token in tokens]
    
    def _call(self, prompt: str, generation_config: Optional[GenerationConfigMixin]) -> str:
        if generation_config is None:
            generation_config = GenerationConfigMixin()
        tokens = self._tokenize(prompt)
        # TODO: investigate. Is that a HF thing?! 
        generation_config["pad_token_id"] = self.tokenizer.eos_token_id
        output_tokens = self.model.generate(tokens, generation_config=generation_config.to_hf_generation_config())
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]


class MistralLocalGptq(MistralLocal):

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
    

class MistralLocalAwq(MistralLocal):

    def _tokenize(self, sequence: List[str] | str) -> BatchEncoding:
        if not isinstance(sequence, str) and not isinstance(sequence, list):
            raise TypeError("sequences must be a string or list of strings")
        return self.tokenizer(sequence, return_tensors="pt").input_ids.to(device="cuda")

    def _call(self, prompt: str, generation_config: Optional[GenerationConfigMixin]) -> str:
        if generation_config is None:
            generation_config = GenerationConfigMixin()
        tokens = self._tokenize(prompt).to(device="cuda")
        generation_config["pad_token_id"] = self.tokenizer.eos_token_id
        output_tokens = self.model.generate(tokens, generation_config=generation_config.to_hf_generation_config())
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]


class MistralLocalFactory(InferenceLLMFactory):

    @classmethod
    def create(
        cls,
        model: MistralLocal,
        config: Optional[AutoConfig] = None
    ) -> InferenceLLM:
        if model is MistralLocals.mistral_7b_instruct:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(MistralLocal(
                AutoModelForCausalLM.from_pretrained,
                model.value,
                model.value,
                params={"device_map": "cuda:0", "torch_dtype": torch.bfloat16},
                config=config
            ))
            return cls.get_instance()
        elif model is MistralLocals.mistral_7b_instruct_gptq:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(MistralLocalGptq(
                AutoGPTQForCausalLM.from_quantized,
                model.value,
                model.value,
                params={
                    "device": "cuda:0",
                    "safetensors": True
                    },
                config=config
            ))
            return cls.get_instance()
        elif model is MistralLocals.mistral_7b_instruct_awq:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(MistralLocalAwq(
                AutoAWQForCausalLM.from_quantized,
                model.value,
                model.value,
                params={
                    "device_map": "sequential", # sequentialy fills gpu 0 to x, AWQ models should be all in GPU 0; 'cuda:0' not supported..
                    "fuse_layers": False,
                    "trust_remote_code": True,
                    "safetensors": True,
                },
                config=config
            ))
            return cls.get_instance()
        elif model is MistralLocals.mistral_7b_instruct_bnb:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(MistralLocal(
                AutoModelForCausalLM.from_pretrained,
                model.value,
                model.value,
                params={"device_map": "cuda:0"},
                config=config
            ))
            return cls.get_instance()
        elif model is MistralLocals.mistral_7b_instruct_flash_attn:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(MistralLocal(
                AutoModelForCausalLM.from_pretrained,
                MistralLocals.mistral_7b_instruct.value,
                MistralLocals.mistral_7b_instruct.value,
                params={
                    "device_map": "cuda:0",
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "flash_attention_2",
                    },
                config=config,
                model_name=model.value
            ))
            return cls.get_instance()
        raise ValueError(f"Model {model} not supported")