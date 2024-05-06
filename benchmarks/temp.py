# MODIFIYING PYTHON PATH
import sys
import os


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path_to_append = '/proj/experiment/'
if path_to_append not in sys.path:
    sys.path.append(path_to_append)
    print("Added path to python path: ", path_to_append)


# IMPORTING LIBRARIES...
from typing import List, Dict
import json
import glob
import re
import time
import gc
import torch    

from src.llms.llama2 import Llama2LocalFactory, Llama2Locals
from src.llms.mistral import MistralLocals, MistralLocalFactory
from src.llms.gemma import GemmaLocals, GemmaLocalFactory

from src.llms.config.generation_config import GenerationConfigMixin
from src.prompts.few_shot import FewShotTemplate
from src.prompts.prompt import PromptTemplate

import benchmarks.data.prompts.prefix.roles as roles
import benchmarks.data.prompts.prefix.instructions as instructions
from benchmarks.run import run_benchmark

start = time.perf_counter()

# SETUP: read in data

## data are the abstracts of the papers
data: List[Dict[str, str]] = []

## reference is the human written literature review
reference: str = ""

json_files = glob.glob("/proj/experiment/benchmarks/data/papers/education/*.json")

json_files = [file for file in json_files if re.search(r'\d', file)]

for file in json_files:
    with open(file) as f:
        json_data = json.load(f)
    data_dict = {"citation_key": json_data["citation_key"], "abstract": json_data["abstract"]}
    data.append(data_dict)

assert len(data) == 10

# SETUP: create prompts
# - create prompt template
# - fill template
# - create final prompt


RESEARCH_ASSISTENT = "[INST] <<SYS>>" + roles.RESEARCH_ASSISTENT
#RESEARCH_ASSISTENT = "<bos><start_of_turn>user\n" + roles.RESEARCH_ASSISTENT
#RESEARCH_ASSISTENT = roles.RESEARCH_ASSISTENT

LITERATURE_REVIEW = instructions.LITERATURE_REVIEW + "<</SYS>>"
#LITERATURE_REVIEW = instructions.LITERATURE_REVIEW

RESEARCH_ASSISTENT_QUESTION = str.join("\n", [RESEARCH_ASSISTENT, LITERATURE_REVIEW, "Research Question: {research_question}"])

prefix = PromptTemplate(
    template=RESEARCH_ASSISTENT_QUESTION ,
    input_variables=["research_question"]
)
suffix = PromptTemplate(
    template="Lets think step by step",
    input_variables=[],
)
examples_prompt = PromptTemplate(
    template="{citation_key}: {abstract}",
    input_variables=["citation_key", "abstract"],
)

final_prompt = FewShotTemplate(
    suffix=suffix,
    prefix=prefix,
    input_variables=["research_question"],
    examples=data,
    examples_prompt=examples_prompt,
    examples_delimiter="\n",
    validation=True,
)
prompt = final_prompt.format(
    research_question="What is the current state of research on using LLMs to automate educational tasks, specifically through the lens of educational tasks, stakeholders, LLMs, and machine-learning tasks, what are the practical challenges of LLMs in automating educational tasks, specifically through the lens of technological readiness, model performance, and model replicability and what are the ethical challenges of LLMs in automating educational tasks, specifically through the lens of system transparency, privacy, equality, and beneficence?"
)

prompt = prompt + "[/INST]"
#prompt = prompt + "<end_of_turn>\n<start_of_turn>model"

# SETUP: load llm
factory = MistralLocalFactory()
model = MistralLocals.mistral_7b_instruct_speculative_decoding

for attempt in range(3):
        try:
            llm = factory.create(model)
        except Exception as e:
            print(f"[ATTEMPT] {attempt}: Failed to create model: {e}")
            if attempt == 2:
                raise e
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            time.sleep(10)
            continue
        break


result =run_benchmark(
    llm=llm,
    do_lm_harness=False,
    prompt=prompt,
    scorer=None,
    reference=None,
    generation_config=GenerationConfigMixin(do_sample=False, max_new_tokens=500),
)

#DO: Save 
with open("/proj/experiment/benchmarks/results/benchmarks.json", "a") as f:
    json_sr = result.model_dump_json(indent=4)
    f.write(json_sr)

end = time.perf_counter() - start
print(f"Done in {end}!")
