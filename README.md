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

Beyond these foundational techniques, the book introduces **original methodologies** like Fair Pruning (bias-aware optimization), combining industry best practices with cutting-edge research. You'll learn to apply all these techniques to open-source models like Llama, Gemma, and Qwen, with hands-on notebooks that most of them run on a free Colab; those that need a more powerful GPU say so in the chapter’s README.

* Surgically optimize model architectures through depth and width pruning
* Recover lost knowledge using targeted distillation techniques
* Specialize models for your specific domain and use case
* Measure and validate every optimization decision 

![alt text](Images/Ch01_F02_DataTailoringPipeline.drawio.png)

*The Rearchitecting Pipeline. The domain-specific dataset guides the calibration of the base model, informs structural optimization decisions, and drives the final specialization through LoRA fine-tuning. A general dataset supports Knowledge Recovery, ensuring the pruned model retains broad capabilities before domain-specific specialization. This dual approach optimizes each phase for the project's specific objectives.*

## 📚 Book Contents

| | Chapter | What you'll learn |
|---|---|---|
| **PART 1: FOUNDATIONS** | | |
| ✅ | 1 · Why rearchitecting LLMs matters | The case for specialized models over generic LLMs |
| ✅ | [2 · An end-to-end rearchitecting project](CH02/) | Full pipeline: prune and recover |
| ✅ | [3 · A blueprint to modern transformers](CH03/) | GLU architectures, attention, and model internals |
| **PART 2: HANDS-ON OPTIMIZATION** | | |
| ✅ | [4 · Building smaller and faster LLMs with depth pruning](CH04/) | Block removal, caputure block importance with python hooks, evaluate pruning |
| ✅ | [5 · Shaping model architectures via width pruning](CH05/) | GLU neuron selection, data-driven pruning strategies |
| ✅ | [6 · Knowledge recovery through distillation](CH06/) | Recovering capability after structural compression |
| ✅ | [7 · Model specialization](CH07/) | LoRA / DoRA fine-tuning and quantization for domain tasks |
| ✅ | [8 · Attention Optimization](CH08/) | KV cache, attention bypass, inference acceleration |
| ✅ | [9 · Dynamic routing with Mixture of Experts](CH09) | Mixture of Experts (MoE) adapted to SLMs.  |
| **PART 3: BEYOND THE BLACK BOX** | | |
| ✅ | [10 · Exploring the Trasformer Black Box](CH10) | Activation analysis and behavioral interpretability |
| ✅ | [11 · Precision pruning for bias](CH11) | Fair pruning: removing demographic bias at neuron level |
| ✅ | [12 · Case study — Replacing one agent call with a specialized SLM in production](CH12) by [Ishita Verma](https://www.linkedin.com/in/ishitaverma17)| Replacing API calls with a specialized SLM |
| ✅ | [13 · Capstone II — Creating a model family for agents](CH13)  | Create 3 different models from a base one |

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

## 🎓 Using this book in a course

Teaching with *Rearchitecting LLMs*? A class, a study group or a company reading group can run the hands-on labs together using a fork of this repository.

1. **Fork once per cohort.** The instructor forks this repo and enables *Settings → General → Features → Discussions* in the fork. Notebooks keep opening from this repository through the Colab badges, so students always run the latest version.
2. **Seed the discussions.** In the fork's *Actions* tab, enable workflows and run **Seed cohort discussions** ([direct link](../../actions/workflows/seed_discussions.yml)). It copies the *Hands-on Discussion* of every chapter from CH02 to CH13 (Chapter 1 has none), with the same labels (`CH06`, `hands-on labs`), so you can filter the fork's discussions exactly as in the link above. It is safe to run more than once.
3. **Students answer inside the thread.** Each student posts one comment following the results template at the end of the discussion (exercise attempted, model, hardware and versions, configurations tried, results, match with the chapter, key insight, link). Numbers that don't match the chapter are as useful as those that do, and failed runs and OOM errors count. Replies are for peer review.
4. **Hardware.** Most labs run on a free Colab T4. Those needing an L4 or A100 say so in the chapter README, and the Chapter 6 challenge can be explored by changing configuration even without an A100. Students with an Apple Silicon Mac can also run most labs locally: see [Running the notebooks on a Mac](#-running-the-notebooks-on-a-mac-apple-silicon).

**Good to know**
- Enabling Actions in the fork also enables this repository's notebook-cleanup workflow, which amends and force-pushes any push that touches `.ipynb` files. Don't push notebooks to `main`; share them as Colab links or Gists. Once the discussions are seeded, you can disable Actions again in the fork's settings.
- A fork of a public repo is public, so submissions are visible to everyone. Tell students before they post.
- Use *Sync fork* to pick up new notebooks and fixes.
- The repo is Apache-2.0: keep the LICENSE and cite the book (see below).

**Running a course with this book?** I'd like to hear about it: open a Discussion in the *Courses* category.

**[→ Explore all Hands-on Labs Discussions](https://github.com/peremartra/Rearchitecting-LLMs/discussions?discussions_q=is%3Aopen+label%3A%22hands-on+labs%22)**

## 🍏 Running the notebooks on a Mac (Apple Silicon)

The notebooks are written for Colab GPUs, but they also run locally on Apple Silicon Macs (M1 or newer) through PyTorch's Metal backend (MPS). Every notebook picks its device in the same order: CUDA if available, then Apple Silicon (MPS), then CPU. GPU calls go through PyTorch's device-generic `torch.accelerator` API, so the same code runs unchanged on Colab and on a Mac; this needs PyTorch 2.10 or newer.

**Setup**

```bash
git clone https://github.com/peremartra/Rearchitecting-LLMs.git
cd Rearchitecting-LLMs
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate datasets lm-eval optipfair peft trl \
    langdetect codecarbon matplotlib seaborn pandas scikit-learn jupyter
brew install wget            # notebooks download utils.py with !wget
jupyter lab
```

Most notebooks pin the versions the book was tested with in their first cell; run it if you want exactly those versions. Llama and Gemma models, and the `Salesforce/xlam-function-calling-60k` dataset used in Chapter 13, are gated on Hugging Face: accept their licences and run `hf auth login` once.

**What to expect**

- macOS 14 (Sonoma) or newer is required for `bfloat16` on MPS.
- GPU memory is the Mac's unified memory, shared with macOS and your other apps. In `bfloat16` the weights alone take about 2 GB per billion parameters (Qwen3-0.6B ≈ 1.2 GB, SmolLM2-1.7B ≈ 3.4 GB, Llama-3.2-3B ≈ 6.4 GB, Llama-3.1-8B ≈ 16 GB), and distillation notebooks keep a teacher and a student in memory at the same time.
- Latency, throughput and energy numbers (CH04, CH05, CH08, Appendices C and D) will not match the T4/L4 figures in the book; compare models against each other on the same machine. CodeCarbon (energy) reads real power only when it can run `sudo powermetrics` without a password prompt; otherwise it falls back to an estimate based on the CPU's rated power.
- `bitsandbytes` 4-bit quantization (CH03, CH07, CH08) works on MPS with the versions the notebooks pin (0.49.2) and newer, and saves the same memory as on CUDA (Llama-3.2-3B: 2.3 GB in 4-bit vs 6.1 GB in fp16). It runs through a generic PyTorch path, though, so generating with a 4-bit model is much slower than on CUDA (under 1 token/s for Llama-3.2-3B).
- Notebooks whose header says **A100** (CH08 NB04, CH13 NB02/NB03) use batch sizes sized for 40–80 GB of GPU memory. On a Mac with less unified memory, lower the per-step batch and raise gradient accumulation by the same factor, which keeps the effective batch unchanged (for example, in CH13 set `DISTILL_BATCH_SIZE = 1` and `ACCUMULATION_STEPS = 16`).

**Status on Apple Silicon**

Tested on an M2 Max with 32 GB (macOS 26.4, PyTorch 2.15). Training runs were shortened (1 epoch, `lm-eval` limited to 20 examples per task) to check that every cell runs, not to reproduce the book's numbers. Times are for those shortened runs.

| Chapter | Notebooks run end to end on a Mac | Notes |
|---|---|---|
| 2 | NB01, NB02 | ~4 min each |
| 3 | NB01 | Run it with the versions its first cell installs: Phi-4-mini's remote code does not load with transformers 5. Loads Llama-3.1-8B in 4-bit (~20 min) |
| 4 | NB01 Cosine Similarity | ~16 min |
| 5 | NB01, NB02, NB03 | NB02's latency and memory sweep takes ~30 min |
| 6 | NB02 | |
| 7 | NB02 (L4), up to the QDoRA evaluation | 4-bit QLoRA and QDoRA training work. Generating with the unmerged 4-bit QDoRA model is slow: its evaluation cell did not finish within 40 min |
| 8 | NB01 (T4), NB03 | NB01's 4-bit cell takes ~13 min (slow 4-bit generation). **NB02 needs an NVIDIA GPU** (vLLM). NB04 (A100) trains a 3B student against a 3B teacher and did not fit in 32 GB even at batch size 2: the Mac swapped heavily and one epoch did not finish in an hour. The Llama-3.1-8B notebooks were not run |
| 9 | NB01 | |
| 10 | NB01, NB01 (optipfair) | Under 1 min |
| 11 | NB01, NB02 | |
| 12 | NB02 | **NB01 needs an NVIDIA GPU** (vLLM) |
| 13 | NB01, NB02, NB03, NB04 | NB02/NB03 with `DISTILL_BATCH_SIZE = 1`, `ACCUMULATION_STEPS = 16` on 32 GB |
| Appendices | B, C, D | D: run-to-run variation at long prompts can be large on a Mac; the notebook's noise-floor check shows when a difference is real |

Notebooks not listed use the same device code as the ones that were run. Reports from Mac users are welcome in the Discussions.

## 🌟 Support This Project

If you find these techniques useful, consider:
- ⭐ Starring this repo to stay updated
- 🔄 Sharing it with your team
- 💬 Opening Discussions with your questions


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
