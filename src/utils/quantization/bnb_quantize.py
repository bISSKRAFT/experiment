# TODO :)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

name = "google/gemma-7b-it"
quant_path = '/modelcache/leos_models/google/gemma-7b-it'

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(name, device_map="cuda:0", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(name)

print("Saving model...")
model.save_pretrained(quant_path)
tokenizer.save_pretrained(quant_path)

print("Done!")