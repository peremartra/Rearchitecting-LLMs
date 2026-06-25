# Chapter 9: Mixture of Experts (MoE)

This directory contains the notebooks for Chapter 9, we move from the core mixture-of-experts idea to an upcycling workflow that reuses a pre-trained clinical expert. Together, the notebooks show how to add specialization while preserving the base model's general behavior and how routing choices affect the final system.

## Notebooks

### MoE Foundations

### 1. [CH09_NB01_MoE.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH09/CH09_NB01_MoE.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH09/CH09_NB01_MoE.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH09/CH09_NB01_MoE.ipynb)
- **LLM**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Dataset**: `oopere/clinical-ner-qdora`
- **Description**: This notebook builds a two-expert MoE adaptation on top of SmolLM2 to preserve general behavior while improving clinical extraction. It trains a router and one domain expert, then compares soft versus hard routing through schema-compliance evaluation and token-level routing analysis.

---

### Upcycling a Pre-trained Expert

### 2. [CH09_NB02_MOE_Upcycling.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH09/CH09_NB02_MOE_Upcycling.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH09/CH09_NB02_MOE_Upcycling.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH09/CH09_NB02_MOE_Upcycling.ipynb)
- **LLM**: `HuggingFaceTB/SmolLM2-1.7B-Instruct` and `oopere/SmolLM2-1.7B-ClinicalNER`
- **Dataset**: `oopere/clinical-ner-qdora`
- **Description**: This notebook reuses the clinical MLP learned in Chapter 7 and transplants it directly as Expert 1. The workflow skips full expert training, focuses on router adaptation, and evaluates general behavior, clinical extraction quality, and routing decisions.