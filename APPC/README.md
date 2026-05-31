# Appendix C: Energy Consumption Evaluation

This appendix provides a practical reference for measuring GPU energy consumption during inference. The notebook demonstrates how to use the `measure_energy_consumption` helper from the OptiPFair library to estimate the energy footprint of any model, enabling direct comparison between pruned and baseline models across the optimization pipeline.

## Notebooks

### 1. [APPC_NB01_examples.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/APPC/APPC_NB01_examples.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/APPC/APPC_NB01_examples.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/APPC/APPC_NB01_examples.ipynb)
- **LLM**: `google/gemma-3-270m`
- **Dataset**: N/A
- **Description**: Shows how to use the `measure_energy_consumption` helper to estimate GPU energy usage during inference. Provides a practical framework for quantifying the energy savings achieved by depth pruning, width pruning, and other structural optimizations. Runs on a T4 GPU.
