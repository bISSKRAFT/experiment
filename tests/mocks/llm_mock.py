from src.models.llms.base import InferenceLLM


class MockLLM(InferenceLLM):
    def __init__(self):
        self.model_name = "mock"
        self.config = {}

    def _call(self, prompt: str) -> str:
        return "call succeeded!"
    
    def _get_prompt_length_in_tokens(self, prompts: str | list[str]) -> list[int]:
        return [len(prompt.split()) for prompt in prompts]
    
    def _get_config(self, checkpoint: str, config: dict) -> dict:
        return {}
    
    def _get_model(self, checkpoint: str, config: dict, compiling: bool = False):
        return None