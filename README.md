# Rearchitecting LLMs. 
**Structural techniques for efficient models**

[![GitHub stars](https://img.shields.io/github/stars/peremartra/Rearchitecting-LLMs?style=social)](https://github.com/peremartra/Rearchitecting-LLMs/stargazers)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![optiPfair Downloads](https://img.shields.io/pypi/dm/optipfair?label=optiPfair%20downloads)](https://pypi.org/project/optipfair/)
---

Rearchitecting LLMs addresses the growing need for AI professionals who understand how LLMs work at a fundamental level—professionals who can create hyper-efficient models tailored to specific data and tasks rather than relying on one-size-fits-all solutions.

<a href="https://hubs.la/Q040tvsK0">
  <img src="Images/linkedin-profile-banner-martra.jpg" alt="alt text" width="100%">
</a>

*This is the official repository for the book [Rearchitecting LLMs - Structural techniques for efficient models](	
https://hubs.la/Q040tvsK0). Although the notebooks include explanations so they can be understood and run individually, the best experience is through the book, which provides more details on the experiments, decisions, papers, and technologies used.*

**[50% Off with code: ***MLMartra*** from 12 to 26](	
https://hubs.la/Q040tvsK0).📢**

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
