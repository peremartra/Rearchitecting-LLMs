# Chapter 8: Attention Optimization — KV Cache and Structural Pruning

This chapter covers two complementary strategies for reducing the memory and compute cost of inference. The first half targets the **KV cache** — one of the largest and most variable consumers of VRAM during generation — and demonstrates how to quantize it using the Hugging Face Transformers `QuantizedCacheConfig` (via Quanto and BitsAndBytes) and the vLLM serving engine (via FP8 cache). The second half moves from runtime memory management to **structural pruning**: following the findings of He et al. (2024) in [*What Matters in Transformers? Not All Attention is Needed*](https://arxiv.org/abs/2406.15786), we identify and permanently remove the least important attention layers from the model architecture, producing a smaller model that requires no custom inference code. The chapter closes with a Knowledge Distillation step that recovers the generation quality lost during pruning. By the end of this chapter, you will be able to apply both approaches — cache quantization and attention pruning — and understand when each one is the right tool for a given hardware and quality constraint.

## Notebooks

### KV Cache Quantization with Hugging Face Transformers

### 1. [CH08_NB01_T4_KVCache_HuggingFace.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB01_T4_KVCache_HuggingFace.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB01_T4_KVCache_HuggingFace.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB01_T4_KVCache_HuggingFace.ipynb)
- **LLM**: `meta-llama/Llama-3.2-3B`
- **Dataset**: N/A (inference benchmark with a fixed long prompt)
- **Description**: Compares three inference configurations on a T4 GPU to isolate the effect of KV cache quantization on VRAM usage: (1) FP16 baseline, (2) 4-bit weight quantization via BitsAndBytes, and (3) FP16 weights with 4-bit KV cache via Quanto. By fixing one variable at a time, the notebook makes clear which lever — weight precision or cache precision — controls which component of the total VRAM budget.

### 2. [CH08_NB01_L4_KVCache_HuggingFace.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB01_L4_KVCache_HuggingFace.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB01_L4_KVCache_HuggingFace.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB01_L4_KVCache_HuggingFace.ipynb)
- **LLM**: `meta-llama/Llama-3.2-3B`
- **Dataset**: N/A (inference benchmark with a fixed long prompt)
- **Description**: Repeats the same three-configuration KV cache quantization comparison on an L4 GPU. On Ampere or newer architectures, the Quanto CUDA extension compiles and runs natively, so the throughput penalty of cache quantization is substantially smaller than on T4. This notebook provides the L4 reference for the chapter's main experiment.

---

### KV Cache Quantization with vLLM

### 3. [CH08_NB02_T4_KVCache_vLLM.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB02_T4_KVCache_vLLM.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB02_T4_KVCache_vLLM.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB02_T4_KVCache_vLLM.ipynb)
- **LLM**: `meta-llama/Llama-3.2-3B`
- **Dataset**: N/A (inference benchmark with a fixed long prompt)
- **Description**: Demonstrates production-grade FP8 KV cache quantization using the vLLM serving engine on a T4 GPU. The notebook runs vLLM in a standalone subprocess to avoid Jupyter stream incompatibilities and compares the number of GPU KV cache blocks and throughput between a standard FP16 cache and an FP8 cache. A single configuration change doubles the number of available KV cache blocks for the same VRAM budget.

### 4. [CH08_NB02_L4_KVCache_vLLM.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB02_L4_KVCache_vLLM.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB02_L4_KVCache_vLLM.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB02_L4_KVCache_vLLM.ipynb)
- **LLM**: `meta-llama/Llama-3.2-3B`
- **Dataset**: N/A (inference benchmark with a fixed long prompt)
- **Description**: Repeats the vLLM FP8 KV cache experiment on an L4 GPU. On Ampere hardware, vLLM's FP8 path uses native CUDA kernels, providing a more representative throughput comparison than T4. This notebook serves as the L4 reference for the vLLM section of the chapter.

---

### Attention Layer Removal

These notebooks implement the importance-based attention pruning method from He et al. (2024). Each layer's contribution is measured as the cosine distance between its input and output (including the residual connection). Layers whose output is nearly identical to their input are physically deleted from the architecture together with their LayerNorm, and their `forward()` is patched to route the hidden state directly to the MLP block. The result is a smaller model that can be saved and reloaded with a single `trust_remote_code=True` flag, requiring no custom inference code.

### 5. [CH08_NB03_Remove_Attention_.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB03_Remove_Attention_.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB03_Remove_Attention_.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB03_Remove_Attention_.ipynb)
- **LLM**: `meta-llama/Llama-3.2-3B`
- **Dataset**: `HuggingFaceTB/cosmopedia` (calibration only, weighted multi-subset sampling)
- **Description**: Scores every attention sublayer by cosine similarity on a calibration set drawn from Cosmopedia, identifies the three least important layers, permanently deletes their `self_attn` and `input_layernorm` submodules, and patches `forward()` to bypass the attention block. The notebook includes architecture inspection before and after removal, a round-trip save/reload test using a custom `PrunedLlamaForCausalLM` class, and benchmark evaluation on arc_easy, winogrande, hellaswag, lambada_openai, and piqa. Targets a T4 GPU (free tier).

### 6. [CH08_NB03_L4_Llama-3.1-8B.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB03_L4_Llama-3.1-8B.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB03_L4_Llama-3.1-8B.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB03_L4_Llama-3.1-8B.ipynb)
- **LLM**: `meta-llama/Llama-3.1-8B`
- **Dataset**: `HuggingFaceTB/cosmopedia` (calibration only, weighted multi-subset sampling)
- **Description**: Applies the same attention layer removal pipeline to the larger Llama-3.1-8B model on an L4 GPU. Scaling to 8B demonstrates that the importance scoring and physical deletion approach generalizes across model sizes without changes to the algorithm.

---

### Knowledge Recovery

### 7. [CH08_NB04_KD.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB04_KD.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB04_KD.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB04_KD.ipynb)
- **LLM**: `oopere/llama-3.2-3b-attn-drop-3` (student) / `meta-llama/Llama-3.2-3B` (teacher)
- **Dataset**: `HuggingFaceTB/cosmopedia` (10,000 samples, weighted multi-subset sampling)
- **Description**: Applies Knowledge Distillation using [OptiPFair](https://github.com/peremartra/optipfair) to recover the generation quality lost after removing three attention layers in NB03. The frozen teacher model guides the pruned student through one training epoch using a combined task and logits loss. The notebook evaluates both models on the full benchmark suite before and after recovery, and includes training loss curves for the distillation run. Requires an A100 GPU for the 3B model at bfloat16.
