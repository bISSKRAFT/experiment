from unittest import mock
from tests.mocks.llm_mock import MockLLM


def test_serilization() -> None:
    mock_llm = MockLLM()
    mock_llm.config = {"test": "test"}
    res = mock_llm.invoke("test")
    json = res.model_dump_json(indent=2)
    print(json)
    assert json != None