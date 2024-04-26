from abc import abstractmethod, ABC
from enum import Enum
import gc
import torch
from src.models.llms.base import InferenceLLM


class InferenceLLMFactory(ABC):

    _INSTANCE = None

    @abstractmethod
    def create(self, model_or_name: str | Enum) -> InferenceLLM:
        pass

    @classmethod   
    def _check_for_instance(cls, model) -> InferenceLLM | None:
        if cls._INSTANCE is not None and cls._INSTANCE.model_name == model.value:
            return cls._INSTANCE
        return None

    @classmethod
    def _flush(cls):
        print("[InferenceLLMFactory]: FLUSHING INSTANCE...")
        cls._INSTANCE = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @classmethod
    def get_instance(cls) -> InferenceLLM:
        return cls._INSTANCE

    @classmethod
    def set_instance(cls, instance: InferenceLLM):
        cls._INSTANCE = instance