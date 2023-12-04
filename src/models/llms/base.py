from abc import ABC, abstractmethod
from typing import List
from src.models.output import GenerationResult
import time



class BaseLLM(ABC):
    """A class to represent a language model"""

    prompt_lengths: List[int]

    @abstractmethod
    def _generate(
            self,
            prompts: List[str],
            callbacks,
    ) -> GenerationResult:
        """Run the LLM on the given input"""

    @abstractmethod
    def _get_prompt_length_in_tokens(self, prompts: str | List[str]) -> List[int]:
        """Get the length of the prompt in tokens"""

    def generate_prompt(
            self,
            prompts: List[str],
            callbacks,
    ) -> GenerationResult:
        """Generate a LLM response from the given input"""
        self.prompt_lengths = self._get_prompt_length_in_tokens(prompts)
        return self._generate(prompts, callbacks)

    def invoke(self,
               prompt: str,
               callbacks = None,
               ) -> GenerationResult:
        """Generate a LLM response from the given input"""
        return self.generate_prompt([prompt], callbacks)
    
    def batch(self,
              prompts: List[str],
              callbacks = None,
              ) -> GenerationResult:
        """Generate LLM response from a batch of prompts"""
        return self.generate_prompt(prompts, callbacks)


class InferenceLLM(BaseLLM, ABC):
    """Interface for inference language models"""

    model_name: str

    config: dict

    @abstractmethod
    def _call(
            self,
            prompt: str
    ) -> str:
        """Run the LLM on the given input"""

    def _generate(
            self,
            prompts: List[str],
            callbacks,
    ) -> GenerationResult:
        """Run the LLM on the given input"""
        generations = []
        mem_report = []
        inference_time = []

        for prompt in prompts:
            start_time = time.perf_counter()
            generations.append(self._call(prompt))
            inference_time.append(time.perf_counter() - start_time)
            if callbacks:
                mem_report.append(callbacks.memory_report())

        organized_mem_report = callbacks.organize_memory_report(mem_report) if callbacks else {}

        return GenerationResult(
            used_model=self.model_name,
            config=self.config,
            prompt_length_in_tokens=self.prompt_lengths,
            generations=generations,
            inference_time=inference_time if inference_time else None,
            vram_alloc_requests=organized_mem_report.get("alloc_requests", None),
            vram_free_requestst=organized_mem_report.get("free_requests", None),
            vram_allocated_mem=organized_mem_report.get("allocated_mem", None),
            vram_active_mem=organized_mem_report.get("active_mem", None),
            vram_inactive_mem=organized_mem_report.get("inactive_mem", None),
            vram_reserved_mem=organized_mem_report.get("reserved_mem", None),
            vram_alloc_retries=organized_mem_report.get("alloc_retries", None),
        )
