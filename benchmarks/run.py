import time
from typing import List, Optional, Tuple
import lm_eval
from lm_eval.models.huggingface import HFLM
import transformers

from src.models.data.base import BaseDatasetMixin
from src.prompts.few_shot import FewShotTemplate
from src.prompts.prompt import PromptTemplate

from src.llms.config.generation_config import GenerationConfigMixin
from src.models.llms.base import InferenceLLM
from src.models.output import BenchmarkResult
from src.models.quality.base import QualityScorerBase
from src.utils.kpis.performance import calculate_tokens_per_second
from src.utils.profiler.memory_profiler import MemoryProfilerCallback



def _create_hf_eval_lm(initalized_pretrained: transformers.PreTrainedModel) -> HFLM:
    return HFLM(pretrained=initalized_pretrained)

# DEPRACTED
# def _flush_eval_lm(lm_ref) -> None:
#     del lm_ref
#     gc.collect()
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()


def _do_benchmark(benchmark: str, llm: HFLM, n_fewshot: int = 1):
    res_dict = {}
    res = _create_benchmark(benchmark, llm, n_fewshot)
    res_dict["results"] = res.get("results")
    res_dict["configs"] = res.get("configs")
    res_dict["n-shot"] = res.get("n-shot")
    res_dict["config"] = res.get("config")
    res_dict["versions"] = res.get("versions")
    return res_dict

def _create_benchmark(task: str, lm: HFLM, n_fewshot: int = 1):
    task_manager = lm_eval.tasks.TaskManager()
    res=lm_eval.simple_evaluate(
        model=lm,
        tasks=task,
        num_fewshot=n_fewshot,
        #use_cache='/proj/experiment/benchmarks/results',
        task_manager=task_manager
    )
    return res

# DEPRECATED
def preprocess_dataset(
        dataset: BaseDatasetMixin,
        prefix: PromptTemplate,
        example_prompt: PromptTemplate,
        suffix: PromptTemplate,
        n_examples: int = 1,
)-> Tuple[List[str], List[str]]:
    """
    Preprocesses a dataset into a list of prompts.
    :param dataset: The dataset to preprocess.
    :param prefix: The prefix to use.
    :param example_prompt: The example prompt to use.
    :param suffix: The suffix to use.
    :return: A list of filled prompts and a list of labels.
    """
    assert prefix.input_variables == []
    assert len(example_prompt.input_variables) == 3
    assert ["context", "candidate"] == suffix.input_variables
    context, candidates, labels = dataset.get_data()
    prompts = []
    return_labels = []
    # create example pairs of 'n'
    for i in range(0, len(context), n_examples):
        if i + n_examples +1 > len(context):
            #TODO: handle the case where the last batch is smaller than 'n'
            pass
        else:
            examples = [
                {
                    example_prompt.input_variables[0]: context[j],
                    example_prompt.input_variables[1]: candidates[j],
                    example_prompt.input_variables[2]: labels[j]
                }
                    for j in range(i,n_examples+i)
                ]
            few_shot_template = FewShotTemplate(
                prefix=prefix,
                suffix=suffix,
                input_variables=["context", "candidate"],
                examples_prompt=example_prompt,
                examples_delimiter="\n\n",
                examples=examples,
                validation=True,
            )
            prompts.append(few_shot_template.format(
                context=context[1+i+n_examples],
                candidate=candidates[1+i+n_examples],
            ))
            return_labels.append(labels[1+i+n_examples])
    assert len(prompts) == len(return_labels)
    return prompts, return_labels
    

    

def run_benchmark(
        llm: InferenceLLM,
        do_lm_harness: bool,
        prompt: str,
        scorer: Optional[QualityScorerBase],
        reference: Optional[str],
        generation_config: GenerationConfigMixin,
) -> BenchmarkResult:
    """
    Runs a benchmark for a given LLM and a given prompt.
    :param llm: The LLM to benchmark.
    :param prompt: The prompt to benchmark.
    :param scorer: The scorer to use.
    :param reference: The reference to use.
    :param generation_config: The generation config to use.
    :return: The benchmark result.
    """
    assert llm is not None

    hellaswag_res = None
    mem_report_hellaswag = None
    arc_res = None
    mem_report_arc = None

    if do_lm_harness:
        print("[LM-HARNESS]: creating lm")
        try: 
            eval_harness_lm = _create_hf_eval_lm(llm.model)
        except Exception as e:
            print(f"[ATTEMPT] Harness: Failed to create model: {e}")
            eval_harness_lm = _create_hf_eval_lm(llm.model.model)

        print("[LM-HARNESS]: running hellaswag")
        callback = MemoryProfilerCallback("lm_harness")
        hellaswag_res = _do_benchmark('hellaswag', eval_harness_lm, 10)
        mem_report_hellaswag = callback.memory_report()
        print("[LM-HARNESS]: finished hellaswag")

        print("[LM-HARNESS]: running arc")
        arc_res = _do_benchmark('ai2_arc', eval_harness_lm, 25)
        mem_report_arc = callback.memory_report()
        print("[LM-HARNESS]: finished arc")


    # caluclate prompt length
    print("Preprocessing...")
    input_prompt_length = llm._get_prompt_length_in_tokens([prompt])[0]
    remaining_tokens = llm.get_model_size() - input_prompt_length
    print("[INPUT LENGTH]:              ",input_prompt_length)
    print("[REMAINING TOKENS]:          ", remaining_tokens)
    print("done.\n\n")

    # calculate Time To First Token
    print("Pre generation...")
    pre_generation_config = GenerationConfigMixin(max_new_tokens=1)
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
    tokens_second = None
    if result.generation_length_in_tokens is not None and result.inference_time is not None and pre_generation_time is not None:
        tokens_second = calculate_tokens_per_second(
                                                generation_lengths=result.generation_length_in_tokens,
                                                pre_generation_time=pre_generation_time,
                                                inference_times=result.inference_time
                                                )
        print("[TOKENS PER SECOND]:         ",tokens_second)
    
    print("GENERATION:          ")
    print(result.generations)

    # LEGACY
    #
    #
    # print("\n\n")
    # print("--- QUALITY METRICS ---")
    # # score result
    # scores = scorer.compute_score(result.generations, reference)
    # scorer.print_scores()

    return BenchmarkResult(
        **result.model_dump(),
        input_prompt_length=input_prompt_length if isinstance(input_prompt_length, List) else [input_prompt_length],
        time_to_first_token=pre_generation_time if isinstance(pre_generation_time, List) else [pre_generation_time],
        tokens_per_second=tokens_second,
        hellaswag_quality=hellaswag_res,
        hellaswag_performance=mem_report_hellaswag,
        arc_quality=arc_res,
        arc_performance=mem_report_arc,
        time_stamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )
    