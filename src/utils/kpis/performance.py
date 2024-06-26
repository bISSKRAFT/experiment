from typing import List

def calculate_tokens_per_second(
        generation_lengths: List[int],
        inference_times: List[float],
        pre_generation_time: float = 0.0,
        input_prompt_length: int = 0,

) -> List[float]:
    """Calculate the number of tokens per second for each generation."""
    return [(tokens - input_prompt_length) / (time - pre_generation_time) for tokens, time in zip(generation_lengths, inference_times)]