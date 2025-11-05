# Rearchitecting LLMs - Chapter 2

## Rearchitecting an LLM: A hands-on introduction

This directory contains the notebooks for Chapter 2, where we perform the first complete cycle of re-architecting a model. Through a practical example, you will learn to apply a structural optimization and recover the lost knowledge to create a more efficient model.

### Contents

The process is divided into two phases, each implemented in its own notebook:

* **[CH02\_NB01\_Depth\_pruning\_evaluation.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH02/CH02_NB01_Depth_pruning_evaluation.ipynb)**: In this notebook, we perform the "surgery" on the model.

    1.  **Establish the Baseline**: We measure the performance of a standard model (`gemma-3-270m`) to have a clear reference point.
    2.  **Apply Depth Pruning**: We remove two complete layers from the model, reducing its size and, therefore, increasing its inference speed.
    3.  **Evaluate the Impact**: We quantify the performance degradation of the model after pruning to understand the cost of the optimization.

* **[CH02\_NB02\_Knowledge\_Recovery.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH02/CH02_NB02_Knowledge_Recovery.ipynb)**: In this second part, we complete the cycle by "healing" the pruned model.

    1.  **Prepare the Models**: We load the original model as the "teacher" and the pruned model as the "student".
    2.  **Apply Knowledge Distillation**: We train the student model to imitate the "reasoning process" of the teacher model, transferring the lost knowledge.
    3.  **Analyze the Final Result**: We compare the final model with the original to demonstrate that we've created a more efficient model (smaller and faster) that retains most of the original performance.

By the end of this chapter, you will have transformed a generic model into a lighter and faster solution, completing your first *model tailoring* cycle from start to finish.
