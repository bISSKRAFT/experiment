# 🚀 Unlocking the Efficiency of Language Models: Balancing Efficiency and Quality in Instruction-Tuned Models

This project is the implemenation of my master thesis. It uses the 🤗-Transformers library to generate text based on a given prompt.

## 🔝 Project Aim

The aim of this project is to provide an experiment structure to efficiently perform performance evaluations of different optimization techqniues used in LLM research.

### 💿 Reproducability

The project mainly uses 🤗-Transformers and subsequent libraries.

| package         | version     |
| --------------- | ----------- |
| torch           | 2.2.1+cu121 |
| transformers    | 4.38.2      |
| auto_gptq       | 0.7.1       |
| autoawq         | 0.2.4       |
| autoawq_kernels | 0.0.6       |

The performance numbers are generated using a single Nvidia A40 GPU.

### ✅ Installing environment
```python
pip install -r requirements.txt
```
