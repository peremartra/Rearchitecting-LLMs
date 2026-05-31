# Appendix B: Capabilities Evaluation with lm-evaluation-harness

This appendix provides a practical reference for running standardized capability benchmarks on your models using the `lm-evaluation-harness` library. The notebook demonstrates how to use the `model_evaluation` helper from the OptiPFair library to measure performance across common benchmarks, making it easy to compare models before and after any structural optimization.

## Notebooks

### 1. [APPB_NB01_examples.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/APPB/APPB_NB01_examples.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/APPB/APPB_NB01_examples.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/APPB/APPB_NB01_examples.ipynb)
- **LLM**: `google/gemma-3-270m`
- **Dataset**: N/A
- **Description**: Shows how to use the `model_evaluation` helper (OptiPFair) to run capability benchmarks (`arc_easy`, `winogrande`, `hellaswag`, `lambada_openai`) via lm-evaluation-harness. Provides a reproducible baseline evaluation workflow for comparing pruned vs. original models on a T4 GPU.

