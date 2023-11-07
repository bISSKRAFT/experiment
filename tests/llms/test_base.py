from ...src.models.llms.base import InferenceLLM


class MockLLM(InferenceLLM):
    def _call(self, prompt: str) -> str:
        return "call succeeded!"


def test_invoke():
    llm = MockLLM()
    output = llm.invoke("mock prompt")
    assert output == "call succeeded!"
