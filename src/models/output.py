from optparse import Option
from typing import Iterable, List, Optional
from pydantic import BaseModel




class GenerationResult(BaseModel):
    """A class to represent the result of a generation"""

    used_model: Optional[str] = None

    config: Optional[dict] = None

    full_generations: List[str]

    generations: List[str]

    generation_config: Optional[dict] = None

    inference_time: Optional[List[float]] = None

    prompt_length_in_tokens: Optional[List[int]] = None

    generation_length_in_tokens: Optional[List[int]] = None

    ram: Optional[List[float]] = None

    vram_alloc_requests: Optional[List[int]] = None

    vram_free_requestst: Optional[List[int]] = None

    vram_allocated_mem: Optional[List[float]] = None

    vram_active_mem: Optional[List[float]] = None

    vram_inactive_mem: Optional[List[float]] = None

    vram_reserved_mem: Optional[List[float]] = None

    vram_alloc_retries: Optional[List[int]] = None


    def print_statistics(self):
        """Print statistics about the generation"""
        print("Generation Statistics:")
        print("Model:", self.used_model)
        print("Prompt Length:", self.prompt_length_in_tokens)
        print("Generation Length:", self.generation_length_in_tokens)
        print("Inference Time:", self.inference_time)
        print("RAM:", self.ram)
        print("VRAM Allocated Requests:", self.vram_alloc_requests)
        print("VRAM Free Requests:", self.vram_free_requestst)
        print("VRAM Allocated Memory:", self.vram_allocated_mem)
        print("VRAM Active Memory:", self.vram_active_mem)
        print("VRAM Inactive Memory:", self.vram_inactive_mem)
        print("VRAM Reserved Memory:", self.vram_reserved_mem)
        print("VRAM Allocation Retries:", self.vram_alloc_retries)
        
    def __eq__(self, compare: object) -> bool:
        """Check if two GenerationResult objects are equal"""
        return self.generations == compare.generations


class BenchmarkResult(GenerationResult):
    """A class to represent the output of a benchmark"""

    input_prompt_length: Optional[List[int]] = None

    remaining_tokens: Optional[List[int]] = None

    time_to_first_token: Optional[List[float]] = None

    tokens_per_second: Optional[List[float]] = None

    scores: Optional[Iterable] = None


