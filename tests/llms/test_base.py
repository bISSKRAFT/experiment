from src.models.llms.base import InferenceLLM


class MockLLM(InferenceLLM):
    def __init__(self):
        self.model_name = "mock"

    def _call(self, prompt: str) -> str:
        return "call succeeded!"
    
    def _get_prompt_length_in_tokens(self, prompts: str | list[str]) -> list[int]:
        return [len(prompt.split()) for prompt in prompts]
    
    def _get_config(self, checkpoint: str, config: dict) -> dict:
        return {}
    
    def _get_model(self, checkpoint: str, config: dict, compiling: bool = False):
        return None


def test_invoke_without_monitoring():
    llm = MockLLM()
    output = llm.invoke("mock prompt")
    assert output.generations[0] == "call succeeded!"
    assert output.inference_time == None
    assert output.used_model == "mock"
    # no gpu is used
    assert output.vram_alloc_requests == None
    assert output.vram_free_requestst == None
    assert output.vram_allocated_mem == None
    assert output.vram_active_mem == None
    assert output.vram_inactive_mem == None
    assert output.vram_reserved_mem == None
    assert output.vram_alloc_retries == None

def test_batch_without_monitoring():
    llm = MockLLM()
    output = llm.batch(["mock prompt 1", "mock prompt 2"])
    assert output.generations[0] == "call succeeded!"
    assert output.generations[1] == "call succeeded!"
    assert output.inference_time == None
    assert output.used_model == "mock"
    # no gpu is used
    assert output.vram_alloc_requests == None
    assert output.vram_free_requestst == None
    assert output.vram_allocated_mem == None
    assert output.vram_active_mem == None
    assert output.vram_inactive_mem == None
    assert output.vram_reserved_mem == None
    assert output.vram_alloc_retries == None

def test():
    t = {}
    print(t.get("alloc_requests", None))
    
