# Chapter 10: Activation Exploration and Mechanistic Inspection

This directory contains the notebooks for Chapter 10, where we open the model's internal states and inspect how predictions emerge across layers. The chapter focuses on activation tracing for a single prompt, from hook placement and tensor capture points to neuron-level visualizations and layer-wise logit-lens decoding. Two notebook variants are provided: one fully standalone implementation and one implementation that reuses OptiPFair utilities.

## Notebooks

### Standalone Activation Exploration

### 1. [CH10_NB01_Activation_Exploration.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH10/CH10_NB01_Activation_Exploration.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH10/CH10_NB01_Activation_Exploration.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH10/CH10_NB01_Activation_Exploration.ipynb)
- **LLM**: `meta-llama/Llama-3.2-1B` (instrumented) and `Qwen/Qwen3.5-0.8B-Base` (architecture-only comparison)
- **Dataset**: N/A (single-prompt activation analysis)
- **Description**: Fully standalone notebook that captures MLP and attention activations with local hook utilities, visualizes token x neuron and layer x neuron heatmaps, makes the GLU gate/up/down flow visible, and applies a logit lens to show how the top prediction resolves across depth.

---

### Activation Exploration with OptiPFair Utilities

### 2. [CH10_NB01_Activation_Exploration_optipfair.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH10/CH10_NB01_Activation_Exploration_optipfair.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH10/CH10_NB01_Activation_Exploration_optipfair.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH10/CH10_NB01_Activation_Exploration_optipfair.ipynb)
- **LLM**: `meta-llama/Llama-3.2-1B` (instrumented) and `Qwen/Qwen3.5-0.8B-Base` (architecture-only comparison)
- **Dataset**: N/A (single-prompt activation analysis)
- **Description**: Equivalent workflow using `optipfair.bias` for activation capture and heatmap visualization. It reproduces the same interpretability pipeline while relying on the package utilities used in earlier chapters.