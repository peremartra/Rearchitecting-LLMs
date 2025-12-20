# Tailoring LLM Architectures - Chapter 3
## Transformer anatomy: knowing what you'll optimize
This directory contains the notebook for Chapter 3, where we lay the essential groundwork for re-architecting. Before a surgeon can operate, they must have a deep understanding of anatomy. This chapter is dedicated to dissecting various LLMs to understand the evolution of their internal structure.

### Contents
The chapter consists of one practical notebook where we perform a comparative analysis of several key architectures:

* **[CH03_NB01_Model_structures.ipynb](https://github.com/peremartra/Tailoring-LLM-Architectures/blob/main/CH03/CH03_NB01_Model_structures.ipynb)**: In this notebook, we dissect a range of models to build a strong mental map of modern LLM anatomy.

* **The Classic Architecture**: We begin by analyzing DistilGPT2 to establish a baseline and understand the classic components of a decoder-only model.

* **The Modern Evolution**: We move to Llama-3.2 to identify the key architectural shifts (GQA, GLU, RoPE) that define the current generation of LLMs.

* **Architectural Optimizations**: We inspect Gemma to see how recent models implement optimizations (MQA) for enhanced efficiency.

* **Exploring Alternatives**: We briefly look at other modern architectures like microsoft/phi-2 to appreciate the diversity in design choices.

* **The Impact of Quantization**: Finally, we analyze a quantized Llama model to see how optimization techniques physically replace and modify the model's layers.

By the end of this chapter, you will have the fundamental knowledge and practical skills to navigate the internal architecture of any modern LLM, preparing you for the surgical techniques in the chapters to come.
