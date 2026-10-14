# Appendix D: Inference Performance Evaluation

This appendix explains how to measure inference performance in a way that lets you trust the comparison between a base model and its optimized versions: latency, throughput, memory, and how to tell a real difference from run-to-run noise. The notebook includes simplified, self-contained versions of the measurement functions used in the practical chapters.

## Notebooks

### 1. [APPD_NB01_examples.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/APPD/APPD_NB01_examples.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/APPD/APPD_NB01_examples.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/APPD/APPD_NB01_examples.ipynb)
- **LLMs**: `oopere/qwen3-0.6b-weather-geo-specialist`, `oopere/qwen3-0.6b-weather-geo-S`
- **Dataset**: WikiText-2 (used only as a source of real tokens for prompts of an exact length)
- **Description**: Builds prompts of exact length, measures latency and throughput with warm-up and synchronization, separates the time to the first token from the total time, records memory, and uses a repeated baseline to measure the noise floor. Runs on a T4 GPU.
