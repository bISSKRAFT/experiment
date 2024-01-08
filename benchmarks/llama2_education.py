# %%

# MODIFIYING PYTHON PATH
import sys

path_to_append = '/proj/experiment/'
if path_to_append not in sys.path:
    sys.path.append(path_to_append)
    print("Added path to python path: ", path_to_append)
# %%

# IMPORTING LIBRARIES...
from typing import List, Dict
import json
import glob
import re    

from src.llms.llama2 import Llama2Local
from src.llms.config.generation_config import GenerationConfigMixin
from src.prompts.few_shot import FewShotTemplate
from src.prompts.prompt import PromptTemplate
from src.utils.profiler.memory_profiler import MemoryProfilerCallback
from src.utils.kpis.performance import calculate_tokens_per_second
from src.quality.rouge_scorer import RougeQualityScorer

import benchmarks.data.prompts.prefix.roles as roles
import benchmarks.data.prompts.prefix.instructions as instructions
from benchmarks.run import run_benchmark
#%%
# SETUP: rouge scorer
rouge_scorer = RougeQualityScorer(rouge_metrics=['rouge1', 'rouge2', 'rougeL'])

# %%
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

# reading in reference

with open("/proj/experiment/benchmarks/data/papers/education/reference.json") as f:
    json_data = json.load(f)

reference = json_data["reference"]

assert len(reference) > 1
# %%
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

# %%
# SETUP: LLM
llm = Llama2Local.factory("meta-llama/Llama-2-7b-chat-hf")
print("[USED LLM]:        ", llm.model_name)

# %%

# %%
llm = Llama2Local.factory(
                    checkpoint="TheBloke/Llama-2-7b-Chat-AWQ",
                    quanitize="awq",
                    )
print("[USED LLM]:        ", llm.model_name)

# %%
run_benchmark(
    llm=llm,
    prompt=prompt,
    scorer=rouge_scorer,
    reference=reference,
    generation_config=GenerationConfigMixin(do_sample=False, max_new_tokens=500),
)







# %% 
# DO: pre calculations

## caluclate prompt length
input_prompt_length = llm._get_prompt_length_in_tokens(prompt)[0]
print("input length: ",input_prompt_length)
print("remaining tokens", llm.get_model_size() - input_prompt_length)

## calculate pre generation time
generation_config = GenerationConfigMixin(max_new_tokens=0)

result = llm.invoke(prompt,generation_config=generation_config)

pre_generation_time = result.inference_time[0]
print("pre generation time: ", pre_generation_time)
# %%
# DO: invoke llm

generation_config = GenerationConfigMixin(do_sample=False, max_new_tokens=510)

result = llm.invoke(prompt,generation_config=generation_config, callbacks=MemoryProfilerCallback("llama2_education"))
# %%
print("generation length: ", result.generation_length_in_tokens)
# TODO: merge into GenerationResult
print("generated string: ", result.generations)
print("full generation time: ", result.inference_time)
# %%
# DO: calcucate performance metrics
if result.generation_length_in_tokens is not None and result.inference_time is not None and pre_generation_time is not None:
    print("tokens/s: ", calculate_tokens_per_second(result.generation_length_in_tokens, input_prompt_length, result.inference_time, pre_generation_time))
#%%
# DO: score result

scores = rouge_scorer.compute_score(result.generations, reference)
print(scores)
# %%
# DO: save result

# %%
# DO: visualize result


# %%
# DO: with different parameters
unscencored = "TheBloke/llama2_7b_chat_uncensored-AWQ"
default = "TheBloke/Llama-2-7b-Chat-AWQ"
llama2_awq = llama2local_factory(
                    checkpoint=default,
                    quanitize="awq",
                    )

# %%

res = llama2_awq.invoke(
    prompt,
    generation_config=GenerationConfigMixin(do_sample=False, max_new_tokens=516),
    callbacks=MemoryProfilerCallback("llama2_education"))

print(res)
# %%
print(res.generations)
res.print_statistics()

# %%
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name_or_path = "TheBloke/Llama-2-7b-Chat-AWQ"

# Load model
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                          trust_remote_code=False, safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)


# %%
prompt = "Tell me about AI"
prompt_template=f'''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{prompt}[/INST]

'''

print("\n\n*** Generate:")

tokens = tokenizer(
    prompt,
    return_tensors='pt'
).input_ids.cuda()

# Generate output
generation_output = model.generate(
    tokens,
    do_sample=False,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_new_tokens=512
)

print("Output: ", tokenizer.decode(generation_output[0]))

# %%
model.config
# %%
