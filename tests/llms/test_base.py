from src.models.llms.base import InferenceLLM
from tests.mocks.llm_mock import MockLLM




def test_invoke_without_monitoring():
    llm = MockLLM()
    output = llm.invoke("mock prompt")
    assert output.generations[0] == "call succeeded!"
    assert len(output.inference_time) == 1
    assert output.inference_time[0] > 0
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
    assert len(output.inference_time) == 2
    assert output.inference_time[0] > 0
    assert output.used_model == "mock"
    # no gpu is used
    assert output.vram_alloc_requests == None
    assert output.vram_free_requestst == None
    assert output.vram_allocated_mem == None
    assert output.vram_active_mem == None
    assert output.vram_inactive_mem == None
    assert output.vram_reserved_mem == None
    assert output.vram_alloc_retries == None
    
