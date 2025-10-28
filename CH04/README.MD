# Rearchitecting LLMs - Chapter 4

## Depth Pruning: Building Smaller and Faster Models

This directory contains the notebooks for Chapter 4, where we dive deep into systematic depth pruning strategies. Building on the hands-on introduction from Chapter 2, you'll now master advanced techniques for identifying which layers to remove and understand the trade-offs between model size, speed, and performance.

### Contents

This chapter explores, at first, static methods to select the layers to remove, then moves to data-driven approaches to depth pruning, moving beyond simple heuristics to intelligent layer selection:

* **[CH04_NB01_Cosine_Similarity.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH04/CH04_NB01_Cosine_Similarity.ipynb)**: In this notebook, we implement advanced layer importance analysis using cosine similarity.

    1.  **Understanding Layer Contributions**: Learn to measure how much each transformer block modifies the input data using cosine similarity between input and output representations.
    2.  **PyTorch Hooks Mastery**: Implement forward hooks to capture intermediate activations and analyze layer behavior during inference.
    3.  **Data-Driven Layer Selection**: Compare layer importance across different datasets (WikiText vs. SMS Spam) to understand how task complexity affects layer utilization.
    4.  **Creating Specialized Models**: Use the [`optipfair`](https://github.com/peremartra/optipfair) library to create pruned models optimized for specific data types by removing the least important layers.
    5.  **Comprehensive Evaluation**: Measure the impact of intelligent pruning on multiple dimensions: perplexity, accuracy benchmarks (arc_easy, winogrande, hellaswag, lambada_openai), inference time, TTFT (Time to First Token), and throughput.

    **Key Insight**: Layers that contribute little to data transformation (high cosine similarity between input/output) are prime candidates for removal, but their importance varies dramatically depending on the task complexity.

By the end of this chapter, you'll understand that depth pruning is not just about removing layers from the end. It's about analyzing your model's behavior with your specific data to identify which layers you can safely remove while maintaining performance for your use case.
