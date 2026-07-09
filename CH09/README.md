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

### 2. [CH09_NB02_MOE_Upcycling_TopK.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH09/CH09_NB02_MOE_Upcycling_TopK.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH09/CH09_NB02_MOE_Upcycling_TopK.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH09/CH09_NB02_MOE_Upcycling_TopK.ipynb)
- **LLM**: `HuggingFaceTB/SmolLM2-1.7B-Instruct` and `oopere/SmolLM2-1.7B-ClinicalNER`
- **Dataset**: `oopere/clinical-ner-qdora`
- **Description**: This notebook reuses the clinical MLP learned in Chapter 7 and transplants it directly as Expert 1. The workflow skips full expert training, focuses on router adaptation, and evaluates general behavior, clinical extraction quality, and routing decisions.

---

### Three-Expert Upcycling Hands-On

### 3. [CH09_NB03_HandsOn_3Expert_MoE.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH09/CH09_NB03_HandsOn_3Expert_MoE.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH09/CH09_NB03_HandsOn_3Expert_MoE.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH09/CH09_NB03_HandsOn_3Expert_MoE.ipynb)
- **LLM**: `HuggingFaceTB/SmolLM2-1.7B-Instruct` and `oopere/SmolLM2-1.7B-ClinicalNER`
- **Dataset**: `oopere/clinical-ner-qdora` and `flytech/python-codes-25k`
- **Description**: This notebook extends the upcycling design from section 9.3 to three experts: Expert 0 keeps the original base weights, Expert 1 carries the clinical specialization from Chapter 7, and Expert 2 is initialized from a copy of the base MLP and trained on Python code from `flytech/python-codes-25k`. Router and Expert 2 tail layers are trained jointly on a mixture of all three domains. The primary tool for evaluating the result is `analyze_routing`, already introduced in sections 9.2.2. The exercises that follow explore the main variables you can adjust once the baseline notebook is running:
	- Training depth and routing stability
	- Expert isolation through selective freezing
	- Two-phase training for cleaner routing