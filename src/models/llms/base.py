from abc import ABC, abstractmethod
from typing import List
from src.models.output import GenerationResult



class BaseLLM(ABC):
    """A class to represent a language model"""

    @abstractmethod
    def _generate(
            self,
            prompts: List[str],
            callbacks,
    ) -> GenerationResult:
        """Run the LLM on the given input"""

    def generate_prompt(
            self,
            prompt: List[str],
            callbacks,
    ) -> GenerationResult:
        """Generate a LLM response from the given input"""
        return self._generate(prompt, callbacks)

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
        if callbacks is None or not callbacks:
            generations = [self._call(prompt) for prompt in prompts]
            return GenerationResult(
                generations=generations,
                ram=None,
                vram=None,
                inference_time=None,
            )
        generations = []
        mem_report = []
        for prompt in prompts:
            generations.append(self._call(prompt))
            mem_report.append(callbacks.memory_report())
        print("MEM REPORT\n\n", mem_report, "END MEM REPORT\n\n")
        # TODO: save memory report
        return GenerationResult(
            generations=generations,
            ram=None,
            vram=None,
            inference_time=None,
        )