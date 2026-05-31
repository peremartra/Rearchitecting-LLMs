# Rearchitecting LLMs. 
**Structural techniques for efficient models**

[![GitHub stars](https://img.shields.io/github/stars/peremartra/Rearchitecting-LLMs?style=social)](https://github.com/peremartra/Rearchitecting-LLMs/stargazers)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Rearchitecting LLMs addresses the growing need for AI professionals who understand how LLMs work at a fundamental level—professionals who can create hyper-efficient models tailored to specific data and tasks rather than relying on one-size-fits-all solutions.

---

<a href="https://hubs.la/Q040tvsK0">
  <img src="Images/linkedin-profile-banner-martra.jpg" alt="alt text" width="100%">
</a>

*This is the official repository for the book [Rearchitecting LLMs - Structural techniques for efficient models](	
https://hubs.la/Q040tvsK0). Although the notebooks include explanations so they can be understood and run individually, the best experience is through the book, which provides more details on the experiments, decisions, papers, and technologies used.*

The industry is shifting away from generic, closed-source models toward open-source alternatives that offer better stability, data privacy, lower operational costs, and competitive differentiation that proprietary APIs cannot provide. However, this transition faces a critical bottleneck: a shortage of engineers equipped with the deep architectural knowledge required to optimize these models effectively.

The book teaches **optimization techniques** for transforming large pre-trained models into efficient Small Language Models (SLMs). These methodologies—including depth pruning, width pruning in GLU architectures, and knowledge distillation—are similar to approaches used by companies like [**Nvidia**](https://arxiv.org/abs/2407.14679) (Minitron family) and [**Mistral**](https://arxiv.org/abs/2601.08584) (Ministral family) to create production-ready model families.

Beyond these foundational techniques, the book introduces **original methodologies** like Fair Pruning (bias-aware optimization) and Adaptive Attention Bypass (dynamic inference), combining industry best practices with cutting-edge research. You'll learn to apply all these techniques to open-source models like Llama, Gemma, and Qwen, with hands-on notebooks that run on Google Colab's free tier.

* Surgically optimize model architectures through depth and width pruning
* Recover lost knowledge using targeted distillation techniques
* Specialize models for your specific domain and use case
* Measure and validate every optimization decision 

![alt text](Images/Ch01_F02_DataTailoringPipeline.drawio.png)

*The Rearchitecting Pipeline. The domain-specific dataset guides the calibration of the base model, informs structural optimization decisions, and drives the final specialization through LoRA fine-tuning. A general dataset supports Knowledge Recovery, ensuring the pruned model retains broad capabilities before domain-specific specialization. This dual approach optimizes each phase for the project's specific objectives.*

## 🧠 Your Interactive Technical Companion: NotebookLM Space

[![Interact with NotebookLM](https://img.shields.io/badge/🤖_NotebookLM-Ask_Anything-FF6B35?style=for-the-badge&logo=google&logoColor=white)](https://notebooklm.google.com/notebook/a059766a-14bf-4d75-8840-b05a79be680e)

**Start experimenting interactively.**

This NotebookLM space contains all the research papers, chapter notebooks, and optiPfair guides in a conversational format. Think of it as your AI-powered technical assistant for the book, which helps you to become an LLM architect. 

**What you can do:**
- **Ask specific questions**: "How does depth pruning work?" or "How many layers can I remove from a 70B model?"
- **Get code snippets**: "Show me the code to reduce the GLU expansion of Llama3"
- **Explore techniques**: Query any pruning, distillation, or optimization method
- **Troubleshoot**: Get help understanding implementation details from the notebooks

Perfect for:
- Quick reference while coding
- Understanding paper implementations
- Exploring techniques before diving into chapters
- Clarifying concepts on the go

**[→ Launch NotebookLM Space](https://notebooklm.google.com/notebook/a059766a-14bf-4d75-8840-b05a79be680e)**

> 💡 **Pro tip**: Use NotebookLM for quick queries and experimentation. For structured, in-depth learning, the book remains your best companion.

Stop being a mere user. It's time to become an architect.

## 🧪 Join the Hands-on Labs Discussions

Every chapter in this book includes a specific Hands-on Lab designed to push your understanding of model architecture. We use GitHub Discussions as our active laboratory to share metrics, architectural insights, doubts, and even Out-of-Memory (OOM) errors.

**[→ Explore all Hands-on Labs Discussions](https://github.com/peremartra/Rearchitecting-LLMs/discussions?discussions_q=is%3Aopen+label%3A%22hands-on+labs%22)**

Jump into the current active challenges:
* **[CH02] Depth Pruning:** How many layers can you remove before the model breaks? Share your accuracy recovery metrics.
* **[CH03] Architectural Blueprints:** Compare GLU implementations and expansion ratios across modern models like Qwen and Gemma.
* **[CH04] Data-Driven Selection:** Share your progressive degradation curves and memory optimization tricks using PyTorch hooks.

Whether you achieved >90% recovery with fewer distillation samples, or you just want to discuss a specific engineering trade-off you observed, share your configuration and results with the community.

## 📓 Notebooks

| Chapter | Notebook | Open |
|---|---|---|
| **CH02** | [CH02_NB01 · Depth Pruning Evaluation](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH02/CH02_NB01_Depth_pruning_evaluation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH02/CH02_NB01_Depth_pruning_evaluation.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH02/CH02_NB01_Depth_pruning_evaluation.ipynb) |
| **CH02** | [CH02_NB02 · Knowledge Recovery](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH02/CH02_NB02_Knowledge_Recovery.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH02/CH02_NB02_Knowledge_Recovery.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH02/CH02_NB02_Knowledge_Recovery.ipynb) |
| **CH03** | [CH03_NB01 · Model Structures](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH03/CH03_NB01_Model_structures.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH03/CH03_NB01_Model_structures.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH03/CH03_NB01_Model_structures.ipynb) |
| **CH04** | [CH04_NB01 · Cosine Similarity](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH04/CH04_NB01_Cosine_Similarity.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH04/CH04_NB01_Cosine_Similarity.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH04/CH04_NB01_Cosine_Similarity.ipynb) |
| **CH05** | [CH05_NB01 · Width Pruning](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH05/CH05_NB01_width_pruning.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH05/CH05_NB01_width_pruning.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH05/CH05_NB01_width_pruning.ipynb) |
| **CH05** | [CH05_NB02 · Data-Driven Pruning (SMS / Wiki)](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH05/CH05_NB02_data_sms_wiki.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH05/CH05_NB02_data_sms_wiki.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH05/CH05_NB02_data_sms_wiki.ipynb) |
| **CH05** | [CH05_NB03 · Bonus](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH05/CH05_NB03_bonus.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH05/CH05_NB03_bonus.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH05/CH05_NB03_bonus.ipynb) |
| **CH06** | [CH06_NB01 · Knowledge Recovery T4](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB01_Knowledge_Recovery_T4.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB01_Knowledge_Recovery_T4.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB01_Knowledge_Recovery_T4.ipynb) |
| **CH06** | [CH06_NB02 · Width Pruned Model Recovery](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB02_Width_Pruned_Model_Recovery.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB02_Width_Pruned_Model_Recovery.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB02_Width_Pruned_Model_Recovery.ipynb) |
| **CH06** | [CH06_NB03 · Hands-On](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB03_Hands_on.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB03_Hands_on.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB03_Hands_on.ipynb) |
| **CH06** | [CH06_NB03 · Hands-On (Qwen3.5 variant)](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB03_Hands_on_qwen3_5.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB03_Hands_on_qwen3_5.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB03_Hands_on_qwen3_5.ipynb) |
| **CH06** | [CH06_EXP01 · Data-Driven Blocks 2K](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP01_DataDriven_Blocks_2K.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP01_DataDriven_Blocks_2K.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP01_DataDriven_Blocks_2K.ipynb) |
| **CH06** | [CH06_EXP02 · Consecutive Blocks 2K](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP02_DataDriven_Consecutive_Blocks_2K.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP02_DataDriven_Consecutive_Blocks_2K.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP02_DataDriven_Consecutive_Blocks_2K.ipynb) |
| **CH06** | [CH06_EXP03 · Last Blocks 2K](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP03_Last_Blocks_2K.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP03_Last_Blocks_2K.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP03_Last_Blocks_2K.ipynb) |
| **CH06** | [CH06_EXP04 · Last Blocks Preservation 2K](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP04_Last_Blocks_Preservation_2K.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP04_Last_Blocks_Preservation_2K.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP04_Last_Blocks_Preservation_2K.ipynb) |
| **CH06** | [CH06_EXP01 · Data-Driven Blocks 15K](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP01_DataDriven_Blocks_15K.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP01_DataDriven_Blocks_15K.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP01_DataDriven_Blocks_15K.ipynb) |
| **CH06** | [CH06_EXP01 · Data-Driven Blocks 40K](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP01_DataDriven_Blocks_40K.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP01_DataDriven_Blocks_40K.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP01_DataDriven_Blocks_40K.ipynb) |
| **CH07** | [CH07_NB01 · From Matrices to Quantization](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB01_From_Matrices_to_Quantization.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB01_From_Matrices_to_Quantization.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB01_From_Matrices_to_Quantization.ipynb) |
| **CH07** | [CH07_NB02 · QLoRA / QDoRA (L4)](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB02_L4_QLoRA_QDoRA.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB02_L4_QLoRA_QDoRA.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB02_L4_QLoRA_QDoRA.ipynb) |
| **CH07** | [CH07_NB02 · QLoRA / QDoRA (T4)](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB02_T4_QLoRA_QDoRA.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB02_T4_QLoRA_QDoRA.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB02_T4_QLoRA_QDoRA.ipynb) |
| **CH07** | [CH07_NB03 · Hands-On (L4)](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB03_L4_Hands_On.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB03_L4_Hands_On.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB03_L4_Hands_On.ipynb) |
| **CH07** | [CH07_NB · Dataset Generator](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB_dataset_generator.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB_dataset_generator.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH07/CH07_NB_dataset_generator.ipynb) |
| **CH08** | [CH08_NB01 · KV Cache HuggingFace (T4)](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB01_T4_KVCache_HuggingFace.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB01_T4_KVCache_HuggingFace.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB01_T4_KVCache_HuggingFace.ipynb) |
| **CH08** | [CH08_NB01 · KV Cache HuggingFace (L4)](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB01_L4_KVCache_HuggingFace.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB01_L4_KVCache_HuggingFace.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB01_L4_KVCache_HuggingFace.ipynb) |
| **CH08** | [CH08_NB02 · KV Cache vLLM (T4)](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB02_T4_KVCache_vLLM.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB02_T4_KVCache_vLLM.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB02_T4_KVCache_vLLM.ipynb) |
| **CH08** | [CH08_NB02 · KV Cache vLLM (L4)](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB02_L4_KVCache_vLLM.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB02_L4_KVCache_vLLM.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB02_L4_KVCache_vLLM.ipynb) |
| **CH08** | [CH08_NB03 · Remove Attention Layers](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB03_Remove_Attention_.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB03_Remove_Attention_.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB03_Remove_Attention_.ipynb) |
| **CH08** | [CH08_NB03 · Remove Attention — Llama-3.1-8B (L4)](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB03_L4_Llama-3.1-8B.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB03_L4_Llama-3.1-8B.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB03_L4_Llama-3.1-8B.ipynb) |
| **CH08** | [CH08_NB04 · Knowledge Distillation](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB04_KD.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB04_KD.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB04_KD.ipynb) |
| **APPB** | [APPB_NB01 · Capabilities Evaluation](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/APPB/APPB_NB01_examples.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/APPB/APPB_NB01_examples.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/APPB/APPB_NB01_examples.ipynb) |
| **APPC** | [APPC_NB01 · Energy Consumption Evaluation](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/APPC/APPC_NB01_examples.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/Rearchitecting-LLMs/blob/main/APPC/APPC_NB01_examples.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/peremartra/Rearchitecting-LLMs/blob/main/APPC/APPC_NB01_examples.ipynb) |

## 🌟 Support This Project

If you find these techniques useful, consider:
- ⭐ Starring this repo to stay updated
- 🔄 Sharing it with your team
- 💬 Opening Discussions with your questions

Every star helps us reach more LLM engineers who can benefit from this work.

## Citation

If you find this repository or the techniques described in the book useful, please cite it as follows:

**APA:**
Martra, P. (2026). *Rearchitecting LLMs: Structural techniques for efficient models*. Manning Publications. ISBN 9781633434332.

**BibTeX:**
```bibtex
@book{martra2026rearchitecting,
  title={Rearchitecting LLMs: Structural techniques for efficient models},
  author={Martra, Pere},
  isbn={9781633434332},
  year={2026},
  publisher={Manning Publications},
  url={[https://www.manning.com/books/rearchitecting-llms](https://www.manning.com/books/rearchitecting-llms)},
  note={Manning Early Access Program (MEAP)}
}
