# Tailoring LLM Architectures. 
**Surgical Optimization Beyond Fine-Tunings**

[![GitHub stars](https://img.shields.io/github/stars/peremartra/Tailoring-LLM-Architectures?style=social)](https://github.com/peremartra/Tailoring-LLM-Architectures/stargazers)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![optiPfair Downloads](https://img.shields.io/pypi/dm/optipfair?label=optiPfair%20downloads)](https://pypi.org/project/optipfair/)
---

Are you using LLMs, or are you deciding how they should work? That's the difference between a user and an architect.
In this repository, you'll learn to go beyond using APIs or Frameworks; you'll learn to redesign the core of language models to make them faster, cheaper, and, above all, smarter for your specific use case.

You'll get to control what happens inside a model to the point where you'll be able to modify unwanted responses by changing just a few of the model's neurons.

This project is an advanced spin-off of the popular [Large Language Model Notebooks Course](https://github.com/peremartra/Large-Language-Model-Notebooks-Course) (+1800 stars), focused exclusively on optimization and re-architecture techniques that will allow you to go beyond traditional fine-tuning.

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

## The Journey: Your Path from User to LLM Architect
Being an LLM architect is a process. This repository is structured to guide you through the journey, from fundamentals to advanced techniques.

**Understanding the Rearchitecting Pipeline**
* The Challenge: Understand the complete workflow to transform a generic model into a specialized one.
* Your Mission: Assimilate the three key phases of the process: Structural Optimization, Knowledge Recovery, and Specialization.
* Key Resource: (Under construction)

**Your First Surgery - Structural Pruning**
* The Challenge: Perform your first structural modification on a model.
* Your Mission: Learn to identify and remove redundant components from an LLM (Depth Pruning) to make it lighter and faster, and then use Knowledge Distillation to "heal" the model and restore its performance.
* Key Resources: [CH02/CH02_NB01_Depth_pruning_evaluation.ipynb](https://github.com/peremartra/Tailoring-LLM-Architectures/blob/main/CH02/CH02_NB01_Depth_pruning_evaluation.ipynb) and [CH02/CH02_NB02_Knowledge_Recovery.ipynb](https://github.com/peremartra/Tailoring-LLM-Architectures/blob/main/CH02/CH02_NB02_Knowledge_Recovery.ipynb)

**From "Black Box" to Model Anatomy**
* The Challenge: Stop seeing LLMs as a magic black box.
* Your Mission: Understand its internal structure, how information flows, and where its capabilities truly reside. You'll start thinking in terms of layers, blocks, modules, and neurons, not just prompts and responses.
* Key Resource: [CH03/CH03_NB01_Model_structures.ipynb](https://github.com/peremartra/Tailoring-LLM-Architectures/blob/main/CH03/CH03_NB01_Model_structures.ipynb)

🆕 **Data-Driven Depth Pruning** 🆕
* The Challenge: Move beyond random layer removal to intelligent, data-driven pruning decisions.
* Your Mission: Master PyTorch hooks to analyze layer importance using cosine similarity. Learn to identify which layers contribute most to your specific task and create specialized models optimized for different data types (complex vs. simple text).
* What You'll Build: Two optimized models—one for complex text (WikiText) and one for simple text (SMS)—each achieving 11-13% speedup while maintaining task-specific performance.
* Key Insight: Layer importance varies dramatically by task complexity. What's essential for deep reasoning may be dead weight for simple classification.
* Key Resource: [CH04/CH04_NB01_Cosine_Similarity.ipynb](https://github.com/peremartra/Tailoring-LLM-Architectures/blob/main/CH04/CH04_NB01_Cosine_Similarity.ipynb)

**Diving deeper into structural optimization.**
* The Challenge: Adapt the model's structure to its mission and the data it will use.
* Your Mission: Implement depth and width pruning techniques on modern architectures.

**Creating a Specialist with LoRA.**
* The Challenge: Create a domain-specialist model.
* Your Mission: Implement specialized fine-tuning with LoRA on SLMs.

**Controlling the Information Flow - Attention Bypass**
* The Challenge: Go beyond static removal and start managing the information flow dynamically.
* Your Mission: Implement bypass mechanisms in the attention layers, allowing the model to selectively "ignore" calculations that are not necessary for certain tasks, achieving a drastic acceleration in inference.

**Mastery - Adaptive Attention & Fair Pruning**
* The Challenge: Build systems that are not only efficient, but also aware and fair.
Your Mission:
* Adaptive Attention Bypass: Create systems that decide in real-time which attention layers to activate, adapting to the complexity of each input.
* Fair Pruning: Analyze the model's internal activations to identify and prune neurons that contribute to unwanted biases, building models that are efficient and ethical from their architecture.

## Your Architect's Toolkit
Inside this repository you'll find the concepts and code to master re-architecture techniques. To apply these techniques in practice and facilitate their use in production, all the methodologies are being consolidated into the [optiPfair](https://github.com/peremartra/optipfair) support library.

Tools you'll master:

* **Structural Analysis**: Visualization and understanding of architectures like LLaMA, Gemma, and Mistral.
* **Structural Pruning**: Depth Pruning (removing layers) and Width Pruning (reducing neurons).
* **Knowledge Recovery**: Knowledge Distillation to re-train optimized models.
* **Advanced Attention Mechanisms**: Implementation of bypass techniques for ultra-fast inference.
* **Activation Analysis**: The foundation for understanding internal behavior and applying Fair Pruning.

## Start Building
The journey begins with the first step. We recommend you start with the notebooks from Chapter 2, 3, and 4 to build a solid foundation before tackling the more advanced techniques:

1. **Chapter 2**: Learn the basics of depth pruning and knowledge recovery
2. **Chapter 3**: Understand modern transformer architectures (Llama, Gemma, Qwen)
3. **Chapter 4**: Master data-driven layer selection using cosine similarity 🆕

Stop being a mere user. It's time to become an architect.

## 🌟 Support This Project

If you find these techniques useful, consider:
- ⭐ Starring this repo to stay updated
- 🔄 Sharing it with your team
- 💬 Opening Discussions with your questions

Every star helps us reach more LLM engineers who can benefit from this work.
