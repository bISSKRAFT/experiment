from optparse import Option
from typing import List, Optional
from pydantic import BaseModel




class GenerationResult(BaseModel):
    """A class to represent the result of a generation"""

    generations: List[str]

    inference_time: Optional[List[float]]

    ram: Optional[List[float]]

    vram: Optional[List[float]]

    def __eq__(self, compare: object) -> bool:
        """Check if two GenerationResult objects are equal"""
        return self.generations == compare.generations
