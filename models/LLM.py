from abc import ABC, abstractmethod
from typing import List
import torch

class InferenceLLM(ABC):
    
    @abstractmethod
    def invoke(self, tokens: torch.Tensor):
        pass

    @abstractmethod
    def batch(self):
        pass