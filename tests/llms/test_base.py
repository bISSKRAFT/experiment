from src.models.llms.base import InferenceLLM


class MockLLM(InferenceLLM):
    def _call(self, prompt: str) -> str:
        return "call succeeded!"


def test_invoke_without_monitoring():
    llm = MockLLM()
    output = llm.invoke("mock prompt")
    assert output.generations[0] == "call succeeded!"
    assert output.inference_time == None
    assert output.vram == None

def test_batch_without_monitoring():
    llm = MockLLM()
    output = llm.batch(["mock prompt 1", "mock prompt 2"])
    assert output.generations[0] == "call succeeded!"
    assert output.generations[1] == "call succeeded!"
    assert output.inference_time == None
    assert output.vram == None
