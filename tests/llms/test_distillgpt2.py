from src.llms.testLLM import Distillgpt2Local
from src.models.output import GenerationResult
from src.utils.async_memory_profiler import MemoryProfiler
import os


# @pytest.mark.parametrize("sequence, expected_result", [
#    ("Hello, my dog is cute", True),
#    ])

def test_invoke():
    model = Distillgpt2Local("distilgpt2")
    result = model.invoke("Hello, my dog is cute", MemoryProfiler("test", gpu_handle=1, pid=os.getpid()))
    assert isinstance(result, GenerationResult)

def test_result_contains_performance_metrics():
    model = Distillgpt2Local("distilgpt2")
    result = model.invoke("Hello, my dog is cute", MemoryProfiler("test", gpu_handle=1, pid=os.getpid()))
    assert isinstance(result, GenerationResult)
    print(result.inference_time, result.vram)
    assert result.inference_time > 0.0
    assert result.vram > 0.0
