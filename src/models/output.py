from optparse import Option
from typing import List, Optional
from pydantic import BaseModel




class GenerationResult(BaseModel):
    """A class to represent the result of a generation"""

    used_model: Optional[str] = None

    config: Optional[dict] = None

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

    def __eq__(self, compare: object) -> bool:
        """Check if two GenerationResult objects are equal"""
        return self.generations == compare.generations
