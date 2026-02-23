# Chapter 4: Depth Pruning: Building Smaller and Faster Models

This directory contains the notebooks for Chapter 4, where we dive deep into systematic depth pruning strategies. Building on the hands-on introduction from Chapter 2, you'll now master advanced techniques for identifying which layers to remove and understand the trade-offs between model size, speed, and performance. By the end of this chapter, you'll understand that depth pruning is not just about removing layers from the end, but about analyzing your model's behavior with your specific data to identify which layers you can safely remove.

## Notebooks

### 1. [CH04_NB01_Cosine_Similarity.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH04/CH04_NB01_Cosine_Similarity.ipynb)
- **LLM**: `Qwen/Qwen3-0.6B`
- **Dataset**: `wikitext` (`wikitext-2-raw-v1`) and `sms_spam`
- **Description**: This notebook explores how to evaluate the contribution of different transformer blocks to the LLM's objective using cosine similarity to guide data-driven pruning. It compares results between Wikipedia texts and SMS spam to demonstrate how different datasets affect layer importance. Measure the impact of pruning on multiple dimensions: perplexity, accuracy benchmarks (arc_easy, winogrande, hellaswag, lambada_openai), inference time, TTFT (Time to First Token), and throughput.
