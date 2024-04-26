from abc import ABC, abstractmethod
from typing import List, Optional
import time
from src.llms.config.generation_config import GenerationConfigMixin

from src.models.output import GenerationResult



class BaseLLM(ABC):
    """A class to represent a language model"""

    model_name: str

    config: dict

    @abstractmethod
    def _generate(
            self,
            prompts: List[str],
            generation_config: Optional[GenerationConfigMixin],
            callbacks,
    ) -> GenerationResult:
        """Run the LLM on the given input"""

    @abstractmethod
    def _get_prompt_length_in_tokens(
                self, 
                prompts: List[str]
        ) -> List[int]:
        """Get the length of the prompt in tokens"""

    def get_model_size(self) -> int:
        """Get the maximum length of the model"""
        max_position_embeddings = self.config.get("max_position_embeddings", 0)
        n_positions = self.config.get("n_positions", 0)
        if max_position_embeddings > 0:
            return max_position_embeddings
        elif n_positions > 0:
            return n_positions
        else:
            return 0

    def generate_prompt(
            self,
            prompts: List[str],
            generation_config: Optional[GenerationConfigMixin],
            callbacks,
    ) -> GenerationResult:
        """Generate a LLM response from the given input"""
        prompt_lengths = self._get_prompt_length_in_tokens(prompts)
        generations = self._generate(prompts, callbacks, generation_config)
        generations.used_model = self.model_name
        generations.config = self.config
        generations.prompt_length_in_tokens = prompt_lengths
        return generations

    def invoke(self,
               prompt: str,
               generation_config: Optional[GenerationConfigMixin] = None,
               callbacks = None,
               ) -> GenerationResult:
        """Generate a LLM response from the given input"""
        return self.generate_prompt([prompt], callbacks, generation_config)
    
    def batch(self,
              prompts: List[str],
              generation_config: Optional[GenerationConfigMixin] = None,
              callbacks = None,
              ) -> GenerationResult:
        """Generate LLM response from a batch of prompts"""
        return self.generate_prompt(prompts, callbacks, generation_config)


class InferenceLLM(BaseLLM, ABC):
    """Interface for inference language models"""

    @abstractmethod
    def _get_config(
                self, 
                checkpoint: str, 
                config: dict
        ) -> dict:
        """Get the model config from the checkpoint"""

    # @abstractmethod
    # def _get_model(
    #             self, 
    #             checkpoint: str, 
    #             config: dict, 
    #             compiling: bool = False
    #     ):
    #     """Get the model from the checkpoint"""

    @abstractmethod
    def _call(
            self,
            prompt: str,
            generation_config: Optional[GenerationConfigMixin],
    ) -> str:
        """Run the LLM on the given input"""

    def _generate(
            self,
            prompts: List[str],
            generation_config: Optional[GenerationConfigMixin],
            callbacks,
    ) -> GenerationResult:
        """Run the LLM on the given input"""
        full_generations = []
        generations = []
        mem_report = []
        inference_time = []

        # TODO: extraction of generation move to GenerationResult (+test)
        for prompt in prompts:
            start_time = time.perf_counter_ns()
            gen = self._call(prompt, generation_config)
            inference_time.append((time.perf_counter_ns() - start_time) / 10**9)
            if callbacks:
                mem_report.append(callbacks.memory_report())
            full_generations.append(gen)
            generations.append(gen.replace(prompt, ""))

        organized_mem_report = callbacks.organize_memory_report(mem_report) if callbacks else {}
        generation_lengths = self._get_prompt_length_in_tokens(generations)

        return GenerationResult(
            full_generations=full_generations,
            generations=generations,
            generation_config=generation_config.generation_config if generation_config else None,
            generation_length_in_tokens=generation_lengths,
            inference_time=inference_time if inference_time else None,
            vram_alloc_requests=organized_mem_report.get("alloc_requests", None),
            vram_free_requestst=organized_mem_report.get("free_requests", None),
            vram_allocated_mem=organized_mem_report.get("allocated_mem", None),
            vram_active_mem=organized_mem_report.get("active_mem", None),
            vram_inactive_mem=organized_mem_report.get("inactive_mem", None),
            vram_reserved_mem=organized_mem_report.get("reserved_mem", None),
            vram_alloc_retries=organized_mem_report.get("alloc_retries", None),
        )
        
    # def _get_current_device(self):
    #     next(self.model.parameters()).device
    # 
    # def _get_memory_footprint(self):
    #     self.model.get_memory_footprint()
