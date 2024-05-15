
from typing import List


def get_generation_str(
        generations: List[str],
        input_prompts: List[str]
) -> List[str]:
    """Extract only the generated string from a list of generations."""
    return [generation.replace(prompt, "") for generation, prompt in zip(generations, input_prompts)]