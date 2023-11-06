from typing import List
from pydantic import BaseModel

class GenerationResult(BaseModel):
    """A class to represent the result of a generation"""

    generations: List[str]

    def __eq__(self, __value: object) -> bool:
        """Check if two GenerationResult objects are equal"""
        return self.generations == __value.generations