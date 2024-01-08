from src.llms.config.generation_config import GenerationConfigMixin
from src.models.llms.base import InferenceLLM
from src.utils.profiler.memory_profiler import MemoryProfilerCallback
from src.llms.llama2 import Llama2Local
from src.models.output import GenerationResult
from transformers import AutoConfig, GenerationConfig



def test_initialization():
    llama2 = Llama2Local(checkpoint="meta-llama/Llama-2-7b-chat-hf")
    assert isinstance(llama2, InferenceLLM)
    assert isinstance(llama2.config, dict)
    assert llama2.model.config.to_dict() == llama2.config
    assert llama2.get_model_size() == 4096

def test_token_counting():
    llama2 = Llama2Local(checkpoint="meta-llama/Llama-2-7b-chat-hf")
    res = llama2._get_prompt_length_in_tokens(["Hello, my dog is cute"])[0]
    assert res > 0

def test_invoke_():
    llama2 = Llama2Local(checkpoint="meta-llama/Llama-2-7b-chat-hf", compiling=True)
    result = llama2.invoke("Hello, my dog is cute", callbacks=MemoryProfilerCallback("test"))
    print(result)
    assert isinstance(result, GenerationResult)

def test_invoke_generation_config():
    llama2 = Llama2Local(checkpoint="meta-llama/Llama-2-7b-chat-hf")
    generation_config = GenerationConfigMixin(max_new_tokens=10, num_beams=1)
    result = llama2.invoke("Hello, my dog is cute", generation_config=generation_config, callbacks=MemoryProfilerCallback("test"))
    print(result)
    assert isinstance(result, GenerationResult)
    assert result.generation_config is not None
    assert result.generation_config["max_new_tokens"] == 10
    assert result.generation_length_in_tokens is not None
    assert result.generation_length_in_tokens[0] == 18

def test_invoke_generation_config_empty():
    llama2 = Llama2Local(checkpoint="meta-llama/Llama-2-7b-chat-hf")
    result = llama2.invoke("Hello, my dog is cute", callbacks=MemoryProfilerCallback("test"))
    print(result)
    assert isinstance(result, GenerationResult)
    assert result.generation_config is None
    assert result.generation_length_in_tokens is not None
    assert result.generation_length_in_tokens[0] > 0

def test_batch():
    llama2 = Llama2Local(checkpoint="meta-llama/Llama-2-7b-chat-hf")
    result = llama2.batch(["Hello, my dog is cute", "Goodbye, and "], callbacks=MemoryProfilerCallback("test"))
    print(result)
    assert isinstance(result, GenerationResult)

def test_invoke_with_config():
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    config.update({"torch_dtype": "bfloat16"})
    llama2 = Llama2Local(checkpoint="meta-llama/Llama-2-7b-chat-hf",config=config, compiling=False)
    assert llama2.model.config.torch_dtype == "bfloat16"
    result = llama2.invoke("Hello, my dog is cute", callbacks=MemoryProfilerCallback("test"))
    print(result)
    assert isinstance(result, GenerationResult)
    assert result.used_model == "meta-llama/Llama-2-7b-chat-hf"
    assert isinstance(result.config, dict)
    assert result.config == llama2.config
    assert isinstance(result.full_generations, list)
    assert isinstance(result.full_generations[0], str)
    assert isinstance(result.generations, list)
    assert isinstance(result.generations[0], str)
