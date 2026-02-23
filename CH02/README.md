# Chapter 2: Rearchitecting an LLM: A hands-on introduction

This directory contains the notebooks for Chapter 2, where we perform the first complete cycle of re-architecting a model. Through a practical example, you will learn to apply a structural optimization and recover the lost knowledge to create a more efficient model. By the end of this chapter, you will have transformed a generic model into a lighter and faster solution, completing your first *model tailoring* cycle from start to finish.

## Notebooks

### 1. [CH02_NB01_Depth_pruning_evaluation.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH02/CH02_NB01_Depth_pruning_evaluation.ipynb)
- **LLM**: `google/gemma-3-270m` (also evaluated with `meta-llama/Llama-3.2-1B`)
- **Dataset**: `lm-eval` benchmarks (`arc_easy`, `winogrande`, `hellaswag`, `lambada_openai`)
- **Description**: This notebook performs the "surgery" on the model: establishing a baseline, applying depth pruning by removing layers, and quantifying the resulting performance degradation to understand the cost of the optimization.

### 2. [CH02_NB02_Knowledge_Recovery.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH02/CH02_NB02_Knowledge_Recovery.ipynb)
- **LLM**: `google/gemma-3-270m` (Teacher and Student)
- **Dataset**: `DKYoon/SlimPajama-6B`
- **Description**: This notebook completes the cycle by "healing" the pruned model using Knowledge Distillation. The student model is trained to imitate the "reasoning process" of the teacher model, transferring the lost knowledge to create a more efficient model that retains most of the original performance.
