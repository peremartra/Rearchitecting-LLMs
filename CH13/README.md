# Chapter 13: Building a Model Family — Specialization, Pruning, and Distillation

This directory contains the notebooks for Chapter 13. Starting from a LoRA fine-tuned tool-calling specialist, the chapter builds a cascaded family of smaller models through depth pruning, width pruning, and knowledge distillation, with each model acting as the teacher for the next.

## Notebooks

### Building the Specialist Teacher

### 1. [CH13_NB01_LoRA_weather_specialist.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH13/CH13_NB01_LoRA_weather_specialist.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH13/CH13_NB01_LoRA_weather_specialist.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH13/CH13_NB01_LoRA_weather_specialist.ipynb)
- **LLM**: `Qwen3-0.6B`
- **Description**: This notebook fine-tunes Qwen3-0.6B with LoRA into `specialist_model`, a weather and geolocation tool-calling specialist. The LoRA adapter is merged before evaluation, and the resulting model becomes the teacher **T** for the pruning and distillation cascade in the next two notebooks.

---

### Cascaded Pruning and Distillation: Model M

### 2. [CH13_NB02_model_M.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH13/CH13_NB02_model_M.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH13/CH13_NB02_model_M.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH13/CH13_NB02_model_M.ipynb)
- **LLM**: Teacher **T** (Qwen3-0.6B specialist) → Model **M**
- **Description**: This notebook derives Model **M** from the teacher **T** using depth pruning followed by width pruning (via `optipfair`), then recovers quality through knowledge distillation from **T**. Model **M** becomes the teacher for the next cascade step.

---

### Cascaded Pruning and Distillation: Model S

### 3. [CH13_NB03_model_S.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH13/CH13_NB03_model_S.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH13/CH13_NB03_model_S.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH13/CH13_NB03_model_S.ipynb)
- **LLM**: Model **M** (teacher) → Model **S**
- **Description**: Second cascade step: Model **S** is pruned from Model **M** (depth + width via `optipfair`) and distilled with **M** as its teacher, reusing the train/test split inherited from **M**'s handoff file.
