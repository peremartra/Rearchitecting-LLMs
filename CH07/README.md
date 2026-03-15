# Chapter 7: Efficient Fine-Tuning for SLMs

This directory contains the notebooks for Chapter 7, where we complete the rearchitecting pipeline with targeted specialization. We focus on parameter-efficient fine-tuning strategies that preserve compactness while improving task behavior. The chapter moves from first-principles intuition to practical QLoRA/QDoRA workflows and closes with an end-to-end hands-on implementation. By the end of this chapter, you will be able to adapt lightweight open models to domain-specific tasks with reproducible and hardware-aware workflows.

## Notebooks

### Foundations & Theory

### 1. [CH07_NB01_From_Matrices_to_Quantization.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB01_From_Matrices_to_Quantization.ipynb)
- **LLM**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Dataset**: N/A (conceptual and mathematical walkthrough)
- **Description**: Introduces the intuition behind low-rank adaptation and quantization-aware fine-tuning. It builds the conceptual bridge from matrix decomposition to practical parameter-efficient adaptation before training.

---

### LoRA & DoRA Techniques

### 2. [CH07_NB02_L4_QLoRA_QDoRA.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB02_L4_QLoRA_QDoRA.ipynb)
- **LLM**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Dataset**: `oopere/clinical-ner-qdora`
- **Description**: Runs QLoRA and QDoRA fine-tuning on an L4 profile to compare schema compliance and benchmark retention. The notebook evaluates how efficient adaptation changes instruction-following quality in a clinical information extraction task.

### 3. [CH07_NB02_T4_QLoRA_QDoRA.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB02_T4_QLoRA_QDoRA.ipynb)
- **LLM**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Dataset**: `oopere/clinical-ner-qdora`
- **Description**: Repeats the QLoRA/QDoRA workflow under T4 constraints with memory-aware settings. It provides a practical reference for reproducing the chapter experiments on more limited GPU environments.

---

### Hands-On Implementation

### 4. [CH07_NB03_L4_Hands_On.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB03_L4_Hands_On.ipynb)
- **LLM**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Dataset**: `oopere/clinical-ner-qdora`
- **Description**: End-to-end practical notebook for Chapter 7. It consolidates dataset loading, baseline evaluation, QLoRA/QDoRA training, and final comparison in a single reproducible workflow.

---

### Utilities & Data Generation

### 5. [CH07_NB_dataset_generator.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB_dataset_generator.ipynb)
- **LLM**: Multiple providers via LiteLLM
- **Dataset**: `oopere/clinical-ner-qdora` (generated and published)
- **Description**: Generates the synthetic Clinical NER dataset used in Chapter 7 and prepares it for publication on Hugging Face. It includes category-balanced generation, schema validation, and documentation-ready metadata for reproducible fine-tuning.
