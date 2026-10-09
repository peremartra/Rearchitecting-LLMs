# Chapter 12: Capstone I — Replacing One Agent Call with a Specialized SLM in Production

This directory contains the notebooks for Chapter 12. They explore two practical parts of replacing an agent's API call with a specialized small language model: serving a structurally pruned model with vLLM, and estimating the effects of width alignment, attention-layer count, and serving costs.

## Notebooks

### Serving an Attention-Free Model with vLLM

### 1. [CH12_NB01_vllm_attention_free.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH12/CH12_NB01_vllm_attention_free.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH12/CH12_NB01_vllm_attention_free.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH12/CH12_NB01_vllm_attention_free.ipynb)
- **LLM**: `meta-llama/Llama-3.2-3B` (unmodified control) and `oopere/llama-3.2-3b-attn-drop-6` (attention removed from 6 of 28 blocks)
- **Dataset**: N/A (fixed prompt for generation and latency tests)
- **Description**: Tests whether vLLM can serve the attention-pruned checkpoint. It compares the standard load path with a custom vLLM model class, then checks generation, allocated KV cache, and batch-1 latency against the control. It does not evaluate model quality. The gated control model requires license acceptance and Hugging Face authentication.

### Padding and KV Cache Arithmetic

### 2. [CH12_NB02_arithmetic.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH12/CH12_NB02_arithmetic.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH12/CH12_NB02_arithmetic.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH12/CH12_NB02_arithmetic.ipynb)
- **LLM**: N/A (pure Python calculations)
- **Dataset**: N/A (no dataset)
- **Description**: Companion to sections 12.7 and 12.9. Demonstrates padding waste from hardware-aligned widths, estimates KV cache size as attention layers are removed, and calculates break-even and per-request costs for GPU hosting versus an API. No GPU or model weights are required; the cost examples are illustrative, not measured traces.
