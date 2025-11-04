# Rearchitecting LLMs - Chapter 5

## Width Pruning in Modern Architectures

This directory contains the notebooks for Chapter 5. After mastering Depth Pruning in the previous chapter, we now delve into a more precise surgery: **Width Pruning**.

In this chapter, you will learn to surgically reduce the size of the MLP modules, a critical component that consumes a large number of parameters in modern models like Llama, Gemma, or Mistral. Instead of removing entire blocks, we will select and eliminate individual neurons within the GLU architecture, creating lighter, faster, and more energy-efficient models.

### Contents

This chapter explores two fundamental strategies for deciding which neurons to remove:

  * **[CH05\_NB01\_width\_pruning.ipynb](https://www.google.com/search?q=https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH05/CH05_NB01_width_pruning.ipynb)**: In this notebook, we implement **static** width pruning, based on weight magnitude.

    1.  **GLU Pruning Anatomy**: We reinforce the knowledge from Chapter 3, understanding why the `gate_proj` and `up_proj` layers must be pruned synchronously.
    2.  **Static Selection (Data-Free)**: We implement a selection strategy that ranks neurons based solely on their weight magnitude, under the hypothesis that smaller weights contribute less.
    3.  **Expansion Reduction**: We apply this technique to `Llama-3.2-1B` to surgically reduce its MLP expansion ratio from 4x to a more efficient 2.4x.
    4.  **Trade-off Analysis**: We evaluate the resulting model and discover a fascinating *trade-off*: while general reasoning (GSM8K) degrades, the ability to follow instructions (IFEval) and truthfulness (TruthfulQA) improve drastically.

  * **[CH05\_NB02\_Data\_Driven\_Selection.ipynb](https://www.google.com/search?q=https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH05/CH05_NB02_Data_Driven_Selection.ipynb)**: (Notebook under development) Here, we evolve to **data-driven** pruning, which is smarter and more specialized.

    1.  **Hybrid Importance**: We learn to go beyond static weights to incorporate the model's **activations**.
    2.  **Capturing Activations**: We revisit PyTorch *hooks* (from Chap 4) to capture the output of the `down_proj` layers and use them as an importance signal.
    3.  **Specialized Pruning**: We use the `optipfair` library to create pruned models that specialize in a specific data type (similar to the Wiki vs. SMS analysis in the previous chapter).
    4.  **Comparative Evaluation**: We compare the performance of statically pruned models versus data-driven pruned models to demonstrate the value of using activations.

**Key Insight**: By the end of this chapter, you will understand that width pruning doesn't just reduce the model's size; it fundamentally alters its behavior. You will learn to use this technique to create smaller models that, paradoxically, can become *better* at specific tasks, like following instructions, by eliminating the "noise" from general-knowledge neurons.
