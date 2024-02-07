from abc import abstractmethod, ABC

from src.models.llms.base import InferenceLLM


class InferenceLLMFactory(ABC):

    @abstractmethod
    def create(self, model_name: str) -> InferenceLLM:
        pass