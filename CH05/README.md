# Chapter 5: Width Pruning in Modern Architectures

This directory contains the notebooks for Chapter 5. After mastering Depth Pruning in the previous chapter, we now delve into a more precise surgery: **Width Pruning**.

In this chapter, you will learn to surgically reduce the size of the MLP modules, a critical component that consumes a large number of parameters in modern models like Llama, Gemma, or Mistral. Instead of removing entire blocks, we will select and eliminate individual neurons within the GLU architecture, creating lighter, faster, and more energy-efficient models. 

By the end of this chapter, you will understand that width pruning doesn't just reduce the model's size; it fundamentally alters its behavior. You will learn to use this technique to create smaller models that, paradoxically, can become *better* at specific tasks, like following instructions, by eliminating the "noise" from general-knowledge neurons.

## Notebooks

### 1. [CH05_NB01_width_pruning.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH05/CH05_NB01_width_pruning.ipynb)
- **LLM**: `meta-llama/Llama-3.2-1B`
- **Dataset**: N/A (Data-free static pruning, evaluated on `GSM8K`, `IFEval`, `TruthfulQA` benchmarks)
- **Description**: This notebook implements static width pruning based on weight magnitude for GLU architecture. It surgically reduces the MLP expansion ratio and analyzes the trade-off in reasoning, instruction following, and truthfulness.

### 2. [CH05_NB02_data_sms_wiki.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH05/CH05_NB02_data_sms_wiki.ipynb)
- **LLM**: `meta-llama/Llama-3.2-1B`
- **Dataset**: `wikitext` (`wikitext-2-raw-v1`) and `sms_spam`
- **Description**: This notebook demonstrates data-driven width pruning by capturing the activations of the `down_proj` layers using PyTorch hooks. It creates two specialized models calibrated on different datasets to evaluate domain-specific specialization.

### 3. [CH05_NB03_bonus.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH05/CH05_NB03_bonus.ipynb)
- **LLM**: `meta-llama/Llama-3.2-1B`
- **Dataset**: `wikitext` (`wikitext-2-raw-v1`) and `sms_spam`
- **Description**: A bonus notebook testing a static pruned 20% model on Wiki2 and SMS Datasets for cross-evaluation insights.
