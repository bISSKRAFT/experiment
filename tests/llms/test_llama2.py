from src.models.llms.base import InferenceLLM
from src.utils.profiler.memory_profiler import MemoryProfilerCallback
from src.llms.llama2 import Llama2Local
from src.models.output import GenerationResult
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM



def test_token_counting():
    llama2 = Llama2Local(checkpoint="meta-llama/Llama-2-7b-chat-hf")
    res = llama2._get_prompt_length_in_tokens(["Hello, my dog is cute"])[0]
    assert res > 0

def test_invoke_():
    llama2 = Llama2Local(checkpoint="meta-llama/Llama-2-7b-chat-hf", compiling=True)
    result = llama2.invoke("Hello, my dog is cute", MemoryProfilerCallback("test"))
    print(result)
    assert isinstance(result, GenerationResult)

def test_batch():
    llama2 = Llama2Local(checkpoint="meta-llama/Llama-2-7b-chat-hf")
    result = llama2.batch(["Hello, my dog is cute", "Goodbye, and "], MemoryProfilerCallback("test"))
    print(result)
    assert isinstance(result, GenerationResult)

def test_invoke_with_config():
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    config.update({"torch_dtype": "bfloat16"})
    llama2 = Llama2Local(checkpoint="meta-llama/Llama-2-7b-chat-hf",config=config, compiling=False)
    assert llama2.model.config.torch_dtype == "bfloat16"
    result = llama2.invoke("Hello, my dog is cute", MemoryProfilerCallback("test"))
    print(result)
    assert isinstance(result, GenerationResult)
    assert result.used_model == "meta-llama/Llama-2-7b-chat-hf"
    assert isinstance(result.config, dict)
    assert result.config == llama2.config
    assert isinstance(result.generations, list)
    assert isinstance(result.generations[0], str)
