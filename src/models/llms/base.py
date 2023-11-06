from abc import ABC, abstractmethod
from typing import List
from ..output import GenerationResult

class BaseLLM(ABC):
    """A class to represent a language model"""

    @abstractmethod
    def _generate(
        self,
        prompts: List[str],
    ) -> GenerationResult:
        """Run the LLM on the given input"""

    def generate_prompt(
            self,
            prompt: List[str],
    ) -> GenerationResult:
        """Generate a prompt from the given input"""
        return self._generate([prompt])
    
    def invoke(self,
               prompt: str,
    ) -> str:
        """Generate a prompt from the given input"""
        return self.generate_prompt(prompt).generations[0]


class InferenceLLM(BaseLLM,ABC):
    
    @abstractmethod
    def _call(
        self,
        prompt: str
    ) -> str:
        """Run the LLM on the given input"""

    def _generate(
        self,
        prompts: List[str],
    ) -> List[str]:
        """Run the LLM on the given input"""
        generations = [self._call(prompt) for prompt in prompts]
        return GenerationResult(generations=generations)