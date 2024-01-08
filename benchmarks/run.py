from typing import Iterable, List, Optional

from src.llms.config.generation_config import GenerationConfigMixin
from src.models.llms.base import InferenceLLM
from src.models.quality.base import QualityScorerBase
from src.utils.kpis.performance import calculate_tokens_per_second
from src.utils.profiler.memory_profiler import MemoryProfilerCallback


def run_benchmark(
        llm: InferenceLLM,
        prompt: str,
        scorer: QualityScorerBase,
        reference: str,
        generation_config: GenerationConfigMixin,
) -> None:
    """
    Runs a benchmark for a given LLM and a given prompt.
    :param llm: The LLM to benchmark.
    :param prompt: The prompt to benchmark.
    :param scorer: The scorer to use.
    :param reference: The reference to use.
    :param generation_config: The generation config to use.
    :return: None
    """
    
    # caluclate prompt length
    print("Preprocessing...")
    input_prompt_length = llm._get_prompt_length_in_tokens([prompt])[0]
    print("[INPUT LENGTH]:              ",input_prompt_length)
    print("[REMAINING TOKENS]:          ", llm.get_model_size() - input_prompt_length)
    print("done.\n\n")

    # calculate pre generation time
    print("Pre generation...")
    pre_generation_config = GenerationConfigMixin(max_new_tokens=0)
    pre_gen_res = llm.invoke(prompt,generation_config=pre_generation_config)
    pre_generation_time = pre_gen_res.inference_time[0]
    print("[PRE GENERATION TIME]:       ", pre_generation_time)
    print("done.\n\n")

    # run generation
    print("Running generation...")
    result = llm.invoke(
        prompt,
        generation_config,
        MemoryProfilerCallback("callbacks"),
    )
    print("done.\n\n")

    print("--- GENERATION RESULT ---")
    print("[INPUT LENGTH]:              ",input_prompt_length)
    print("[GENERATION LENGTH]:         ", result.generation_length_in_tokens)
    print("[FULL GENERATION TIME]:      ", result.inference_time)

    print("\n\n")
    print("--- PERFORMANCE METRICS ---")
    print("[VRAM ACTIVE MEMORY]:        ",[result / 1024 / 1024 for result in result.vram_active_mem] if result.vram_active_mem is not None else None)
    print("[VRAM RESERVED MEMORY]:      ",[result / 1024 / 1024 for result in result.vram_reserved_mem] if result.vram_reserved_mem is not None else None)
    if result.generation_length_in_tokens is not None and result.inference_time is not None and pre_generation_time is not None:
        print("[TOKENS PER SECOND]:         ", calculate_tokens_per_second(
                                                generation_lengths=result.generation_length_in_tokens,
                                                pre_generation_time=pre_generation_time,
                                                inference_times=result.inference_time
                                                )
            )
    
    print("GENERATION:          ")
    print(result.generations)

    print("\n\n")
    print("--- QUALITY METRICS ---")
    # score result
    scores = scorer.compute_score(result.generations, reference)
    scorer.print_scores()
    