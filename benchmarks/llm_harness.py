# %%
!source /proj/experiment/pipenv/bin/activate
# %%
# DO: HellaSwag
!lm_eval --model hf --model_args pretrained="meta-llama/Llama-2-7b-chat-hf" --tasks hellaswag --num_fewshot 10 --device cuda:0 --batch_size auto --output_path /proj/experiment/benchmarks/results --use_cache /proj/experiment/benchmarks/results
# %%
# DO: ARCDataset
!lm_eval --model hf --model_args pretrained="meta-llama/Llama-2-7b-chat-hf" --tasks ai2_arc --num_fewshot 25 --device cuda:0 --batch_size auto --output_path /proj/experiment/benchmarks/results --use_cache /proj/experiment/benchmarks/results
# %%
# DO: Hellaswag: llama2 7b the bloke awq 
!lm_eval --model hf --model_args pretrained="TheBloke/Llama-2-7b-Chat-AWQ" --tasks hellaswag --num_fewshot 10 --device cuda:0 --batch_size auto --output_path /proj/experiment/benchmarks/results/llama27b_thebloke_awq.json --use_cache /proj/experiment/benchmarks/results

# %%
# DO: ARCDataset: llama2 7b the bloke awq
!lm_eval --model hf --model_args pretrained="TheBloke/Llama-2-7b-Chat-AWQ" --tasks ai2_arc --num_fewshot 25 --device cuda:0 --batch_size auto --output_path /proj/experiment/benchmarks/results/llama27b_thebloke_awq_arc.json --use_cache /proj/experiment/benchmarks/results