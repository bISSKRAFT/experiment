# MODIFIYING PYTHON PATH
import sys

path_to_append = '/proj/experiment/'
if path_to_append not in sys.path:
    sys.path.append(path_to_append)
    print("Added path to python path: ", path_to_append)


# IMPORTING LIBRARIES...
from typing import List, Dict
import json
import glob
import re    

from src.llms.llama2 import Llama2LocalFactory, Llama2Locals
from src.llms.config.generation_config import GenerationConfigMixin
from src.prompts.few_shot import FewShotTemplate
from src.prompts.prompt import PromptTemplate
from src.utils.profiler.memory_profiler import MemoryProfilerCallback
from src.utils.kpis.performance import calculate_tokens_per_second
from src.quality.rouge_scorer import RougeQualityScorer
from src.data.hf_datasets import HellaSwagDataset

import benchmarks.data.prompts.prefix.roles as roles
import benchmarks.data.prompts.prefix.instructions as instructions
from benchmarks.run import run_benchmark, preprocess_dataset

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

LITERATURE_REVIEW = instructions.LITERATURE_REVIEW + "<</SYS>>"

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
print(prompt)

# SETUP: load factory
llama2_factory = Llama2LocalFactory()


result =run_benchmark(
    llm_factory=llama2_factory,
    model=Llama2Locals.llama_2_7b_chat_awq,
    prompt=prompt,
    scorer=None,
    reference=None,
    generation_config=GenerationConfigMixin(do_sample=False, max_new_tokens=500),
)

# DO: Save 
with open("/proj/experiment/benchmarks/results/benchmarks.json", "a") as f:
    json_sr = result.model_dump_json(indent=4)
    f.write(json_sr)
