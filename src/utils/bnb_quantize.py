# TODO :)

from transformers import AutoModelForCausalLM, AutoTokenizer

name = "meta-llama/Llama-2-13b-chat-hf"
quant_path = "/modelcache/leos_models/meta-llama/Llama-2-13b-chat-hf-bnb"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(name, device_map="cuda:2", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(name)

print("Saving model...")
model.save_pretrained(quant_path)
tokenizer.save_pretrained(quant_path)

print("Done!")