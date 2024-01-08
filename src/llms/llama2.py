import gc
from typing import Any, Callable, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BatchEncoding
from awq import AutoAWQForCausalLM
from src.llms.config.generation_config import GenerationConfigMixin
from src.models.llms.base import InferenceLLM
import torch


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
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)
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
        tokens = self._tokenize(prompt).to(device="cuda:0")
        output_tokens = self.model.generate(tokens, generation_config=generation_config.to_hf_generation_config())
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    def _tokenize(self, sequence: List[str] | str) -> BatchEncoding:
        if not isinstance(sequence, str) and not isinstance(sequence, list):
            raise TypeError("sequences must be a string or list of strings")
        return self.tokenizer(sequence, return_tensors="pt").input_ids.cuda()
    
    def _get_prompt_length_in_tokens(self, prompts: List[str] | str) -> List[int]:
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self._tokenize(prompts)
        return [len(token) for token in tokens]
    

    @classmethod
    def factory(cls,
            checkpoint: str = "",
            config: Optional[AutoConfig] = None,
            quanitize: Optional[str] = None,
        ) -> 'Llama2Local':
        # TODO: 
        # - qunatization: awq and gtpq
        # - fusing: with awq quantization
        # - flash_anttention: with quantization and without
        """Factory for constructing Llama2Local"""
        #global cls.__INSTANCE
        if cls.__INSTANCE is not None and cls.__INSTANCE.model_name == checkpoint:
            return cls.__INSTANCE
        del cls.__INSTANCE
        gc.collect()
        if not checkpoint:
            raise ValueError("checkpoint must be specified")
        if quanitize:
            if quanitize not in ["gptq", "awq"]:
                raise ValueError(f"quanitize must be one of gptq, but was {quanitize}.")
            if "thebloke" not in checkpoint.lower():
                raise ValueError(f"quanitize via checkpoint is only supported for TheBloke models, but was {checkpoint}.")
            cls.__INSTANCE = Llama2Local(
                AutoAWQForCausalLM.from_quantized,
                checkpoint,
                params={
                    "fuse_layers": True,
                    "trust_remote_code": False,
                    "safetensors": True,
                },
                config=config
            )
            return cls.__INSTANCE
        cls.__INSTANCE = Llama2Local(
                AutoModelForCausalLM.from_pretrained,
                checkpoint,
                params={"device_map": "cuda:0"},
                config=config
            )
        return cls.__INSTANCE
    


# class Llama2Optimum(InferenceLLM):

#     def __init__(
#             self,
#             checkpoint: str
#         ) -> None:
#         pass
        
    