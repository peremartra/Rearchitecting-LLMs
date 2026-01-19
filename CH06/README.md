# Chapter 6: Knowledge Distillation — Information Retrieval Pipeline

This chapter covers the **Knowledge Recovery** stage of the pipeline. The technique we will use is **Knowledge Distillation**, but the process begins with a fundamental decision: **which parts of the model should we remove?**

Selecting the right layers or Transformer blocks to prune is crucial, as it directly impacts how much knowledge is lost and how effectively it can be recovered through distillation.

---

## Experiments Overview

The following notebooks document different experiments exploring various strategies for layer/block removal before applying Knowledge Distillation:

### Initial Experiments (2K samples)

| Notebook | Strategy | Description |
|----------|----------|-------------|
| `CH06_NB_EXP01_DataDriven_Blocks_2K.ipynb` | **Data-Driven Block Selection** | Uses a data-driven approach to identify and remove the least important Transformer blocks based on their contribution to model performance. |
| `CH06_NB_EXP02_DataDriven_Consecutive_Blocks_2K.ipynb` | **Data-Driven Consecutive Blocks** | Similar to EXP01, but constraints the removal to consecutive Transformer blocks. |
| `CH06_NB_EXP03_Last_Blocks_2K.ipynb` | **Last Blocks Removal** | A simpler heuristic approach that removes the last N blocks of the model, based on the assumption that later layers contain more task-specific knowledge. |
| `CH06_NB_EXP04_Last_Blocks_Preservation_2K.ipynb` | **Last Blocks Preservation** | Explores preserving specific final layers while removing intermediate ones, testing if critical output representations need protection. |

---

## Best Performing Approach

The experiment that achieved the **best performance without any knowledge recovery** was:

> **`CH06_NB_EXP01_DataDriven_Blocks_2K`** (Data-Driven Block Selection)

This data-driven approach for selecting which Transformer blocks to remove proved most effective at maintaining model capabilities even before applying distillation techniques.

---

## Extended Experiments

Based on the success of the data-driven Transformer block selection strategy, the experimentation continues with larger training datasets:

| Notebook | Training Samples | Purpose |
|----------|------------------|---------|
| `CH06_NB_EXP01_DataDriven_Blocks_15K.ipynb` | 15,000 | Scaling up training data to evaluate if more samples improve knowledge recovery. |
| `CH06_NB_EXP01_DataDriven_Blocks_40K.ipynb` | 40,000 | Further scaling to determine the relationship between dataset size and recovery effectiveness. |

---

## Results

All experimental results are stored in the `results/` directory as JSON files for detailed analysis and comparison.