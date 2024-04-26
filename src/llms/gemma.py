from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BatchEncoding
from typing import Any, Callable, Dict, List, Optional
import torch
from awq import AutoAWQForCausalLM
from auto_gptq import AutoGPTQForCausalLM


from src.models.llms.factories import InferenceLLMFactory
from src.llms.config.generation_config import GenerationConfigMixin
from src.models.llms.base import InferenceLLM

class GemmaLocals(Enum):
    gemma_7b_it = "google/gemma-7b-it"
    gemma_7b_it_bnb = "/modelcache/leos_models/google/gemma-7b-it"
    gemma_7b_it_awq = "/modelcache/leos_models/google/gemma-7b-it-awq"
    gemma_7b_it_gptq = "/modelcache/leos_models/google/gemma-7b-it-gptq"
    gemma_7b_it_flash_att = "google/gemma-7b-it-flash-att"
    gemma_7b_it_speculative_decoding = "google/gemma-7b-it-speculative-decoding"

class GemmaLocal(InferenceLLM):
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
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=False)[0]
    
    def _tokenize(self, sequence: List[str] | str) -> BatchEncoding:
        if not isinstance(sequence, str) and not isinstance(sequence, list):
            raise TypeError("sequences must be a string or list of strings")
        return self.tokenizer(sequence, return_tensors="pt").input_ids.to(device="cuda:0")
    
    def _get_prompt_length_in_tokens(self, prompts: List[str] | str) -> List[int]:
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self._tokenize(prompts)
        return [len(token) for token in tokens]
    

class GemmaLocalAwq(GemmaLocal):

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
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=False)[0]

class GemmaLocalGptq(GemmaLocal):

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
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=False)[0]
    
    def _get_prompt_length_in_tokens(self, prompts: List[str] | str) -> List[int]:
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self._tokenize(prompts).input_ids
        return [len(token) for token in tokens]


class GemmaLocalSpeculativeDecoding(GemmaLocal):

    def __init__(self,
                 factory: Callable, 
                 checkpoint_model: str,
                 checkpoint_tokenizer: str,
                 config: Optional[AutoConfig] = None, 
                 params: Optional[Dict[str, Any]] = None,
                 model_name: Optional[str] = None):
        super().__init__(factory, checkpoint_model, checkpoint_tokenizer, config, params, model_name)
        support_model_ckpt = "google/gemma-2b-it"
        self.support_model = AutoModelForCausalLM.from_pretrained(support_model_ckpt, device_map="auto", torch_dtype=torch.bfloat16)
        self.config["support_model"] = support_model_ckpt
        print("Constructor: ", self.__class__, "Support model: ", support_model_ckpt)
    
    def _tokenize(self, sequence: List[str] | str) -> BatchEncoding:
        if not isinstance(sequence, str) and not isinstance(sequence, list):
            raise TypeError("sequences must be a string or list of strings")
        return self.tokenizer(sequence, return_tensors="pt")
    
    def _call(self, prompt: str, generation_config: Optional[GenerationConfigMixin]) -> str:
        if generation_config is None:
            generation_config = GenerationConfigMixin()
        generation_config["pad_token_id"] = self.tokenizer.eos_token_id
        tokens = self._tokenize(prompt).to("cuda")
        output_tokens = self.model.generate(
            **tokens, 
            assistant_model=self.support_model, 
            generation_config=generation_config.to_hf_generation_config())
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=False)[0]
    
    def _get_prompt_length_in_tokens(self, prompts: List[str] | str) -> List[int]:
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self._tokenize(prompts).input_ids
        return [len(token) for token in tokens]

class GemmaLocalFactory(InferenceLLMFactory):

    @classmethod
    def create(
        cls,
        model: GemmaLocals,
        config: Optional[AutoConfig] = None,
    ) -> InferenceLLM:
        if model == GemmaLocals.gemma_7b_it:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(GemmaLocal(
                factory=AutoModelForCausalLM.from_pretrained,
                checkpoint_model=model.value,
                checkpoint_tokenizer=model.value,
                config=config,
                params={
                    "device_map": "cuda:0",
                    "torch_dtype": torch.bfloat16,
                    },
                model_name=model.value
            
            ))
            return cls.get_instance()
        if model in [GemmaLocals.gemma_7b_it_bnb]:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(GemmaLocal(
                factory=AutoModelForCausalLM.from_pretrained,
                checkpoint_model=model.value,
                checkpoint_tokenizer=model.value,
                config=config,
                params={
                    "device_map": "cuda:0",
                    },
                model_name=model.value
            ))
        if model == GemmaLocals.gemma_7b_it_awq:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(GemmaLocalAwq(
                factory=AutoAWQForCausalLM.from_quantized,
                checkpoint_model=model.value,
                checkpoint_tokenizer=model.value,
                config=config,
                params={
                    "device_map": "sequential", # sequentialy fills gpu 0 to x, AWQ models should be all in GPU 0; 'cuda:0' not supported..
                    "fuse_layers": False,
                    "trust_remote_code": True,
                    "safetensors": True,
                },
                model_name=model.value
            ))
        if model == GemmaLocals.gemma_7b_it_gptq:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(GemmaLocalGptq(
                AutoGPTQForCausalLM.from_quantized,
                model.value,
                GemmaLocals.gemma_7b_it.value,
                params={
                    "device": "cuda:0",
                    "trust_remote_code": True,
                    "safetensors": True,
                    #"disable_exllamav2": True
                },
                config=config
            ))
            return cls.get_instance()
        if model == GemmaLocals.gemma_7b_it_flash_att:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(GemmaLocal(
                factory=AutoModelForCausalLM.from_pretrained,
                checkpoint_model=GemmaLocals.gemma_7b_it.value,
                checkpoint_tokenizer=GemmaLocals.gemma_7b_it.value,
                config=config,
                params={
                    "device_map": "cuda:0",
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "flash_attention_2",
                    },
                model_name=model.value
            ))
            return cls.get_instance()
        if model == GemmaLocals.gemma_7b_it_speculative_decoding:
            instance_check = cls._check_for_instance(model)
            if instance_check:
                return instance_check
            cls._flush()
            cls.set_instance(GemmaLocalSpeculativeDecoding(
                factory=AutoModelForCausalLM.from_pretrained,
                checkpoint_model=GemmaLocals.gemma_7b_it.value,
                checkpoint_tokenizer=GemmaLocals.gemma_7b_it.value,
                config=config,
                params={
                    "device_map": "cuda:0",
                    "torch_dtype": torch.bfloat16,
                    },
                model_name=model.value
            ))
            return cls.get_instance()
        raise ValueError(f"Model {model} is not supported by GemmaLocalFactory")