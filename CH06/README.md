# Chapter 6: Knowledge Distillation — Information Recovery Pipeline

This chapter covers the **Knowledge Recovery** stage of our pipeline. The main technique we use is **Knowledge Distillation**, but the process begins with a crucial decision: which parts of the model should we target? Selecting the right layers or Transformer blocks to prune directly impacts the knowledge loss and the subsequent effectiveness of distillation-based recovery. This chapter features main approaches to knowledge recovery as well as a rich set of experiments outlining various dataset-driven layer selection strategies.

## Notebooks

### Main Knowledge Recovery

### 1. [CH06_NB01_Knowledge_Recovery_T4.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB01_Knowledge_Recovery_T4.ipynb)
- **LLM**: `google/gemma-3-270m`
- **Dataset**: `HuggingFaceTB/cosmopedia`
- **Description**: Re-evaluates Knowledge Distillation from Chapter 2 using a different, high-quality dataset (Cosmopedia) to observe its impact on knowledge recovery.

### 2. [CH06_NB02_Width_Pruned_Model_Recovery.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB02_Width_Pruned_Model_Recovery.ipynb)
- **LLM**: `Qwen/Qwen3-0.6B`
- **Dataset**: `HuggingFaceTB/cosmopedia`
- **Description**: Applies Knowledge Distillation to recover performance from a model that has undergone Width Pruning, contrasting recovery patterns with Depth Pruning.

### 3. [CH06_NB03_Hands_on.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB03_Hands_on.ipynb)
- **LLM**: `google/gemma-3-270m`
- **Dataset**: `HuggingFaceTB/cosmopedia`
- **Description**: A hands-on summary notebook implementing the full Knowledge Recovery process.

---

### Layer Selection Strategies (2K Samples)

These notebooks explore strategies for layer/block removal *before* applying Knowledge Distillation:

### 4. [CH06_NB_EXP01_DataDriven_Blocks_2K.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP01_DataDriven_Blocks_2K.ipynb)
- **LLM**: `google/gemma-3-270m`
- **Dataset**: `HuggingFaceTB/cosmopedia`
- **Description**: Uses a data-driven approach to identify and remove the least important Transformer blocks based on their contribution to model performance. *(This approach achieved the best performance without recovery).*

### 5. [CH06_NB_EXP02_DataDriven_Consecutive_Blocks_2K.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP02_DataDriven_Consecutive_Blocks_2K.ipynb)
- **LLM**: `google/gemma-3-270m`
- **Dataset**: `HuggingFaceTB/cosmopedia`
- **Description**: Constraints the data-driven removal to consecutive Transformer blocks.

### 6. [CH06_NB_EXP03_Last_Blocks_2K.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP03_Last_Blocks_2K.ipynb)
- **LLM**: `google/gemma-3-270m`
- **Dataset**: `HuggingFaceTB/cosmopedia`
- **Description**: A heuristic approach that removes the last N blocks, assuming later layers contain task-specific rather than fundamental knowledge.

### 7. [CH06_NB_EXP04_Last_Blocks_Preservation_2K.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP04_Last_Blocks_Preservation_2K.ipynb)
- **LLM**: `google/gemma-3-270m`
- **Dataset**: `HuggingFaceTB/cosmopedia`
- **Description**: Explores preserving specific final layers while removing intermediate ones, testing if critical output representations need protection.

---

### Scaling Data-Driven Pruning

Based on the success of EXP01, we scaled up the experimental training datasets to measure recovery effectiveness.

### 8. [CH06_NB_EXP01_DataDriven_Blocks_15K.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP01_DataDriven_Blocks_15K.ipynb)
- **LLM**: `google/gemma-3-270m`
- **Dataset**: `HuggingFaceTB/cosmopedia` (15,000 samples)
- **Description**: Scales up the data-driven block selection to 15,000 samples to evaluate if more samples improve knowledge recovery.

### 9. [CH06_NB_EXP01_DataDriven_Blocks_40K.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH06/CH06_NB_EXP01_DataDriven_Blocks_40K.ipynb)
- **LLM**: `google/gemma-3-270m`
- **Dataset**: `HuggingFaceTB/cosmopedia` (40,000 samples)
- **Description**: Further scales to 40,000 samples to determine the strict relationship between dataset size and recovery effectiveness.