from src.utils.profiler.memory_profiler import MemoryProfilerCallback
from src.llms.llama2 import Llama2Local
from src.models.output import GenerationResult
import torch
import os




def test_gpu_connection():
    assert torch.cuda.is_available() == True
    assert torch.cuda.device_count() > 0
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

def test_invoke_():
    llama2 = Llama2Local(checkpoint="meta-llama/Llama-2-7b-chat-hf")
    result = llama2.invoke("Hello, my dog is cute", MemoryProfilerCallback("test"))
    print(result)
    print(result.inference_time, result.vram)
    assert isinstance(result, GenerationResult)

def test_batch():
    llama2 = Llama2Local(checkpoint="meta-llama/Llama-2-7b-chat-hf")
    result = llama2.batch(["Hello, my dog is cute", "Goodbye, and "], MemoryProfilerCallback("test"))
    print(result)
    print(result.inference_time, result.vram)
    assert isinstance(result, GenerationResult)
