# Chapter 11: Precision pruning for bias

This directory contains the notebooks for Chapter 11, where we move from broad model modification to targeted neuron-level interventions for reducing bias. The notebooks use contrastive activation analysis to identify neurons associated with racial and gender bias, scale selected neurons without changing the model architecture, and then validate the intervention on the BBQ benchmark.

## Notebooks

### Developing a Signed Bias Intervention

### 1. [CH11_NB01_Signed_Bias_Intervention.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH11/CH11_NB01_Signed_Bias_Intervention.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH11/CH11_NB01_Signed_Bias_Intervention.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH11/CH11_NB01_Signed_Bias_Intervention.ipynb)
- **LLM**: `meta-llama/Llama-3.2-1B`
- **Dataset**: N/A (hand-crafted contrastive prompt pairs)
- **Description**: This notebook develops a fairness-aware neuron intervention from first principles. It compares minimally different prompts, captures MLP activations, and uses a signed activation difference to locate neurons associated with racial and gender bias. The highest-scoring neurons are selectively scaled through their `up_proj` weights, then inspected through before-and-after generations, layer-wise heatmaps, logit-lens analysis, and optional capability benchmarks. The resulting race intervention is carried forward to the BBQ evaluation in the next notebook.

---

### Validating Bias Reduction with BBQ

### 2. [CH11_NB02_BBQ_Benchmark.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH11/CH11_NB02_BBQ_Benchmark.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH11/CH11_NB02_BBQ_Benchmark.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH11/CH11_NB02_BBQ_Benchmark.ipynb)
- **LLM**: `meta-llama/Llama-3.2-1B`
- **Dataset**: `oskarvanderwal/bbq` (test split)
- **Description**: This notebook reuses the five-neuron race intervention identified in NB01 and evaluates it with the BBQ bias benchmark. It sets up a baseline-versus-intervened comparison in ambiguous and disambiguated race/ethnicity contexts, reporting accuracy and bias-score changes across a 25,000-example evaluation subset. The baseline run can be enabled when a fresh comparison is needed, while the default notebook avoids that extra evaluation cost. The benchmark provides a quantitative test of whether the targeted intervention moves model behavior closer to neutral while preserving the original model and intervention setup.