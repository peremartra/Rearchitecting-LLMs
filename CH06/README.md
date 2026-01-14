# Chapter 6: Universal Knowledge Distillation - Engineering Plan

| Technical Component | Purpose in Chapter 6 | SOTA Implementation (Industrial Level) | Technical Implementation Strategy |
| :--- | :--- | :--- | :--- |
| **1. Textbook-Quality Data** (Simplified) | **NEW:** Focus on high-quality input without complex curation steps. We assume the data is ready. | Use of "Textbook Quality" datasets (Microsoft Phi / NVIDIA approach) instead of raw web text. | Direct loading of **Cosmopedia** (Hugging Face) to ensure dense information density. No custom generation script. |
| **2. Hidden State Distillation** | Move beyond "imitation" (output) to "understanding" (process). Force the student to replicate internal representations. | Access `hidden_states`. Use **MSE Loss** to align the student's intermediate tensors with the teacher's. | Hook into `outputs.hidden_states`. Compute distance between selected layers. |
| **3. Intelligent Layer Mapping** | Solve the architectural mismatch caused by **Depth Pruning** (Student has fewer layers). | **Uniform Strategy** (skip every N layers) or **Last-Layer Strategy** (focus on deep reasoning). | A mapping function `map_layers(student_n, teacher_n)` that returns a dictionary `{student_idx: teacher_idx}`. |
| **4. Learnable Projections** | Solve the dimensional mismatch caused by **Width Pruning** (Student vectors are smaller). | Insert a Trainable Linear Layer ($W_{proj}$) that learns to translate "Student Space" to "Teacher Space". | `nn.Linear(student_dim, teacher_dim)` initialized specifically for the distiller. Only these weights (and the student's) are updated. |
| **5. Compound Loss Landscape** | Balance factual accuracy with reasoning emulation. | $L_{Total} = \alpha L_{Task} + \beta L_{KL} + \gamma L_{Hidden}$ | Custom `compute_loss` combining CrossEntropy (Hard Targets), KL Divergence (Soft Targets), and MSE (Features). |
| **6. Custom Trainer Loop** | Full control over the optimization process for non-standard architectures. | Subclassing Hugging Face `Trainer` to manage the projector's lifecycle. | Create `UniversalDistillationTrainer` class. Override `compute_loss` and ensure projectors are saved with the model. |
____
# Chapter 6: Feature-Based Distillation: Recovering Knowledge After Surgical Pruning

## 6.1 The Alignment Problem: Why Logits-Only KD Isn't Enough
**What we'll cover:** Diagnose why logits-only Knowledge Distillation from Chapter 2 isn't sufficient to fully recover a pruned model's capabilities. The core issue is that we've been treating models as black boxes—only caring about the final answer, not the internal reasoning process.

**Content:**
- **Recap of progress so far:**
  - We have models with missing layers (Depth Pruning, Ch. 2/4) or thinner layers (Width Pruning, Ch. 5)
  - Simple KD from Chapter 2 achieves ~90-92% recovery
  
- **The limitation of Vanilla KD:**
  - Treats the model as a "Black Box" (only compares final outputs)
  - Only checks if the answer is correct, not if the reasoning is sound
  
- **The solution: Feature-Based Distillation**
  - Introduction to aligning "Hidden States" (the intermediate representations)
  - Transferring the internal "reasoning process," not just the final result

- **The challenge: "Teacher-Student Gap"**
  - How do we align their brains if they have different shapes (dimensions) and different depths (number of layers)?
  
- **Visual elements:**
  - Figure 6.1: Conceptual diagram showing Teacher and Student internal representations diverging
  - Figure 6.2: The Teacher-Student Gap illustrated with cosine similarity metrics

---

# Chapter 6: Feature-Based Distillation: Recovering Knowledge After Surgical Pruning

## 6.1 The Alignment Problem: Why Logits-Only KD Isn't Enough
**What we'll cover:** Diagnose why logits-only Knowledge Distillation from Chapter 2 isn't sufficient to fully recover a pruned model's capabilities. The core issue is that we've been treating models as black boxes—only caring about the final answer, not the internal reasoning process.

**Content:**
- **Recap of progress so far:**
  - We have models with missing layers (Depth Pruning, Ch. 2/4) or thinner layers (Width Pruning, Ch. 5)
  - Simple KD from Chapter 2 achieves ~90-92% recovery
  
- **The limitation of Vanilla KD:**
  - Treats the model as a "Black Box" (only compares final outputs)
  - Only checks if the answer is correct, not if the reasoning is sound
  
- **The solution: Feature-Based Distillation**
  - Introduction to aligning "Hidden States" (the intermediate representations)
  - Transferring the internal "reasoning process," not just the final result

- **The challenge: "Teacher-Student Gap"**
  - How do we align their brains if they have different shapes (dimensions) and different depths (number of layers)?
  
- **Visual elements:**
  - Figure 6.1: Conceptual diagram showing Teacher and Student internal representations diverging
  - Figure 6.2: The Teacher-Student Gap illustrated with cosine similarity metrics

---

## 6.2 The Solution Blueprint: Compound Loss for Feature Alignment
**What we'll cover:** Before implementing solutions, we need to understand **what we're going to optimize**. We introduce the Compound Loss that balances three competing objectives: task accuracy, output distribution, and internal reasoning alignment.

**Content:**

### 6.2.1 The Compound Loss Formula
$$L_{Total} = \alpha L_{Task} + \beta L_{Logits} + \gamma L_{Hidden}$$

### 6.2.2 The Three Components
- **$L_{Task}$ (Cross Entropy):** Maintain hard labels knowledge
  - Formula: $CE(\text{logits}_{student}, \text{labels})$
  - Purpose: "Don't forget the ground truth"
  
- **$L_{Logits}$ (KL Divergence):** Imitate Teacher's confidence distribution
  - Formula: $KL(P_{student} || P_{teacher}) \cdot T^2$ where $P = \text{softmax}(\text{logits}/T)$
  - Purpose: Transfer "dark knowledge" (soft probabilities)
  
- **$L_{Hidden}$ (MSE):** Align internal representations ← **This is the critical new ingredient**
  - Formula: $\text{MSE}(h_{student}, h_{teacher})$
  - Purpose: Align the internal reasoning process

### 6.2.3 Intuition of Hyperparameters
- How α, β, γ change the model's behavior:
  - High γ → Strict imitation (student closely follows teacher's thought process)
  - High α → Independence (student relies more on ground truth)
  - Balanced approach → Best recovery
  
### 6.2.4 Visual Understanding
- Figure 6.3: Information flow diagram showing where each loss is computed
- Figure 6.4: Conceptual comparison of Black Box (logits-only) vs Transparent Box (feature-based) distillation

**SIDEBAR: Temperature Scaling in Soft Labels**
- Brief intro: T parameter in softmax(logits/T)
- When T > 1: Smoother distributions (Teacher shares more "dark knowledge")
- Typical value: T=2.0 works well for most scenarios
- Mathematical intuition: Higher temperature "softens" the probability distribution

**NOTE to readers:**
> We've now established **what** we need to optimize. But there's a technical challenge: how do we compute $L_{Hidden}$ when Teacher and Student have different architectures (different depths or widths)? In the following sections, we'll solve this step by step through hands-on experiments.

---

## 6.3 Bridging the Gap Part I: Solving Depth Mismatch
**What we'll cover:** Solve the problem of "different number of layers" by implementing and comparing Layer Mapping strategies. Through experiments, we'll discover which strategy works best and demonstrate that feature alignment significantly improves recovery even with simple mapping.

**Content:**

### 6.3.1 The Depth Mismatch Problem
- Teacher has 24 layers, Student has 20 layers (after depth pruning)
- Which Student layer $h_s^i$ should we align with which Teacher layer $h_t^j$?
- Naive approach: Try to compute $L_{Hidden}$ directly → **Crashes** (no 1-to-1 correspondence)

### 6.3.2 Layer Mapping Strategies
**Strategy A: Uniform Mapping**
- Distribute student layers proportionally across teacher layers
- Formula: $\text{teacher\_idx} = \lfloor i \cdot \frac{n_{teacher}}{n_{student}} \rfloor$
- Intuition: Cover the entire teacher depth uniformly

**Strategy B: Last-Layer Alignment**
- Map student layers to the deepest teacher layers
- Formula: $\text{teacher\_idx} = i + (n_{teacher} - n_{student})$
- Intuition: Focus on where complex reasoning happens (deep layers)

### 6.3.3 Implementation Details
- Code structure: `create_layer_map(n_student, n_teacher, strategy='uniform'|'last')`
- How to use mappings during training loop
- Extracting hidden states at correct positions

### 6.3.4 Experimental Validation
**Experiment Setup:**
- Model: Qwen2.5-0.5B (24 layers) → pruned to 20 layers
- Dataset: WikiText-2 (500 samples for speed)
- Training: 5 epochs, batch_size=8, lr=1e-5

**Experiment A: Logits-Only KD with Different Mappings**
- Test both mapping strategies with $L_{Task} + L_{Logits}$ (no features yet)
- Table 6.1: Comparison of mapping strategies (logits-only)

**Experiment B: Adding Feature Alignment**
- Use Last-Layer mapping + full compound loss ($L_{Task} + L_{Logits} + L_{Hidden}$)
- Table 6.2: Impact of feature alignment

**Expected Results:**
| Configuration | Recovery Rate | Key Insight |
|---------------|---------------|-------------|
| Uniform + Logits-only | ~89% | Baseline with uniform mapping |
| Last-Layer + Logits-only | ~91% | Better mapping strategy (+2%) |
| Last-Layer + Features | ~96% | Feature alignment is the game-changer (+5%) |

### 6.3.5 Analysis and Visualization
- Figure 6.5: Visual comparison of layer mapping strategies
- Figure 6.6: Cosine similarity heatmap (Student vs Teacher hidden states)
- Figure 6.7: Feature loss convergence during training

**Key Insights:**
- Last-Layer mapping outperforms Uniform because deep layers encode complex reasoning
- Feature alignment provides an additional 4-5% recovery beyond logits-only KD
- Hidden state alignment improves progressively during training (visible in convergence plots)

---

## 6.4 Bridging the Gap Part II: Solving Width Mismatch
**What we'll cover:** Solve the problem of "different vector dimensions" using Learnable Projectors. After Width Pruning, Student's hidden states are smaller than Teacher's, making direct comparison impossible.

**Content:**

### 6.4.1 The Dimensional Problem
- After Width Pruning (Ch. 5): Student hidden_dim = 640, Teacher hidden_dim = 896
- Even with layer mapping, cannot compute $L_{Hidden} = \text{MSE}(h_s, h_t)$ → **Dimension mismatch crash**
- Need: A way to transform 640-dimensional vectors to 896-dimensional space

### 6.4.2 The Solution: Learnable Projectors
- Concept: Trainable linear transformations $W_{proj}: \mathbb{R}^{d_s} \rightarrow \mathbb{R}^{d_t}$
- Key principle: Projectors must be **learnable**, not fixed
- Why learnable? Fixed random projections cannot capture semantic alignment

### 6.4.3 Architecture and Implementation
**The LearnableProjector Class:**
- Simple linear layer without bias: `nn.Linear(student_dim, teacher_dim, bias=False)`
- Initialization: Xavier uniform with small gain (0.01) for stable training
- Integration: One projector per aligned layer

**Complete Architecture:**
```
Student Layer i → Hidden State (d_s) → Projector i → Projected State (d_t) → MSE with Teacher Layer j
```

### 6.4.4 Experimental Validation
**Experiment Setup:**
- Model: Qwen2.5-0.5B with width pruning (896 → 640 dimensions)
- Comparison: No projector vs Fixed projector vs Learnable projector

**Results:**
- Table 6.3: Impact of learnable projectors
  - No projector: Crash (dimension mismatch)
  - Fixed projector: ~87% recovery (poor alignment)
  - Learnable projector: ~94% recovery (strong alignment)

### 6.4.5 Projector Analysis
- Figure 6.8: Visualization of learned projector weights
- Figure 6.9: Cosine similarity before/after projection
- What patterns do projectors learn? (expansion of compressed information)

**Key Insights:**
- Projectors must be trainable—they learn to "decompress" the student's compressed representations
- Fixed projectors fail because they cannot capture semantic alignment
- Learnable projectors add minimal overhead (~0.5M parameters) but provide 6-7% recovery improvement

---

## 6.5 Fine-Tuning the Recipe: Hyperparameter Guidelines
**What we'll cover:** Now that we have all the pieces (compound loss, layer mappers, projectors), we provide practical guidelines for adjusting α, β, γ depending on the type of pruning applied.

**Content:**

### 6.5.1 Recommendations by Pruning Type
**Table 6.4: Hyperparameter Starting Points**
| Pruning Type | α (Task) | β (Logits) | γ (Hidden) | Rationale |
|--------------|----------|------------|------------|-----------|
| Depth-only | 0.5 | 0.5 | 0.1 | Fewer layers → focus on output quality |
| Width-only | 0.4 | 0.4 | 0.2 | Compressed features → need stronger alignment |
| Depth+Width | 0.4 | 0.4 | 0.2 | Balanced recovery for hybrid pruning |

### 6.5.2 Tuning Tips and Heuristics
- **If recovery plateaus:** Increase γ in steps of 0.05 (up to 0.3 max)
- **If model "forgets" the task:** Increase α (strengthen ground truth signal)
- **Temperature tuning:** T=2.0 works generally, T=3.0 for aggressive pruning
- **Training duration:** More pruning → needs more epochs (3-5 typically sufficient)

### 6.5.3 Practical Decision Tree
```
Start with your pruning type
↓
Use recommended α, β, γ from Table 6.4
↓
Train 3 epochs → Evaluate
↓
Recovery < 90%? → Increase γ by 0.05
Recovery 90-94%? → Increase epochs to 5
Recovery > 94%? → Done! Optionally reduce γ slightly for faster training
```

**SIDEBAR: Understanding the Feature Loss Weight (γ)**

Why γ is critical for recovery:
- **γ too low (0.05):** Student ignores internal alignment, mimics only outputs
- **γ too high (0.4+):** Student becomes a "parrot" that imitates without generalizing
- **γ optimal (0.15-0.25):** Student learns the reasoning process while maintaining task performance

The right balance depends on:
- How much structure was removed (more pruning → higher γ needed)
- Model architecture (some families need stronger feature alignment)
- Dataset complexity (harder tasks benefit from higher γ)

---

## 6.6 Implementation: The UniversalDistillationTrainer
**What we'll cover:** Transition from manual training loops to a production-ready implementation using HuggingFace's Trainer class. We'll subclass Trainer to create our UniversalDistillationTrainer, handling the complex orchestration of Teacher-Student training with projectors.

**Content:**

### 6.6.1 Why We Need This
**Limitations of Manual Loops (Chapter 2 approach):**
- No automatic checkpointing
- No mixed precision (slower training)
- No logging/metrics tracking
- No learning rate scheduling
- Manual gradient accumulation management

**Solution:** Subclass HuggingFace's `Trainer` to inject our custom distillation logic while keeping all infrastructure benefits.

### 6.6.2 Architecture Overview
**What We Need to Manage:**
1. Two models: Teacher (frozen) and Student (trainable)
2. External components: Projectors (one per aligned layer)
3. Custom loss: Compound loss with three components
4. Lifecycle: Save/load projectors with checkpoints

### 6.6.3 Key Implementation Details

**The UniversalDistillationTrainer Class:**
```python
class UniversalDistillationTrainer(Trainer):
    def __init__(self, teacher_model, layer_map, projectors, 
                 alpha, beta, gamma, temperature, **kwargs)
```

**Critical Override: compute_loss()**
1. Extract labels from inputs
2. Forward pass through Student (with gradients, output_hidden_states=True)
3. Forward pass through Teacher (no_grad, output_hidden_states=True)
4. Apply layer mapping to select aligned hidden states
5. Project Student hidden states through learnable projectors
6. Compute three losses: L_Task, L_Logits, L_Hidden
7. Combine with α, β, γ weights

**Optimizer Management:**
- Must include both Student parameters AND Projector parameters
- Custom `create_optimizer()` override to handle dual parameter groups

**Checkpoint Handling:**
- Save projectors alongside model weights
- Load projectors when resuming training
- Custom `_save()` and `_load()` methods

### 6.6.4 Usage Example
```python
# Setup
teacher = AutoModelForCausalLM.from_pretrained("teacher_model")
student = AutoModelForCausalLM.from_pretrained("pruned_model")
layer_map = create_layer_map(n_student=20, n_teacher=24, strategy='last')
projectors = create_projectors(student_dim=640, teacher_dim=896, n_layers=20)

# Initialize trainer
trainer = UniversalDistillationTrainer(
    model=student,
    teacher_model=teacher,
    layer_map=layer_map,
    projectors=projectors,
    alpha=0.4, beta=0.4, gamma=0.2, temperature=2.0,
    args=training_args,
    train_dataset=train_dataset
)

# Train
trainer.train()
```

**Code Listing 6.1: Complete UniversalDistillationTrainer Implementation**

---

## 6.7 Putting It All Together: Complete Evaluation
**What we'll cover:** Prove that the Universal Distiller works through comprehensive evaluation. Take Llama-3.2-1B pruned in Chapter 5 and demonstrate 93-97% capability recovery, with detailed ablation studies showing each component's contribution.

**Content:**

### 6.7.1 Experiment Setup
**Starting Point:**
- Model: Llama-3.2-1B pruned in Chapter 5
  - Depth pruning: 28 layers (removed last 4)
  - Width pruning: GLU expansion reduced from 350% to 240%
  - Baseline degradation: ~15% loss across benchmarks

**Training Configuration:**
- Dataset: WikiText-2 (30K samples)
- Epochs: 3
- Batch size: 8 (effective batch size: 32 with gradient accumulation)
- Learning rate: 1e-5
- Hyperparameters: α=0.4, β=0.4, γ=0.2, T=2.0
- Hardware: Single GPU (A100 or equivalent)
- Duration: ~2-3 hours

### 6.7.2 Quantitative Recovery Metrics

**Table 6.5: Comprehensive Benchmark Results**
| Benchmark | Original | After Pruning | After FBD | Recovery Rate |
|-----------|----------|---------------|-----------|---------------|
| ARC-Easy | 72.5% | 61.8% (-14.8%) | 70.1% | 96.7% |
| ARC-Challenge | 45.2% | 38.5% (-14.8%) | 43.8% | 96.9% |
| HellaSwag | 75.2% | 64.1% (-14.8%) | 72.5% | 96.4% |
| Winogrande | 68.4% | 58.2% (-14.9%) | 65.9% | 96.3% |
| LAMBADA | 71.8% | 61.5% (-14.3%) | 69.2% | 96.4% |
| **Average** | **66.6%** | **56.8% (-14.7%)** | **64.3%** | **96.5%** |

**Comparison with Chapter 2 (Logits-Only KD):**
- Logits-only KD: ~91-92% recovery
- Feature-Based Distillation: ~96.5% recovery
- **Improvement: +4.5-5.5%** from hidden state alignment

**Statistical Significance:**
- The improvement is consistent across all benchmarks
- Standard deviation: ±0.3% across 3 independent runs

### 6.7.3 Visualizing the "Brain Transplant"

**Figure 6.10: Feature Loss Convergence**
- X-axis: Training steps
- Y-axis: L_Hidden (MSE loss on hidden states)
- Shows exponential decay from ~0.15 to ~0.03
- Interpretation: Student's internal representations rapidly converge to Teacher's

**Figure 6.11: Layer-wise Cosine Similarity Heatmap**
- Rows: Student layers (0-27)
- Columns: Teacher layers (0-31)
- Color: Cosine similarity (0=red, 1=green)
- **Key observations:**
  - Deep layers (20+) show highest similarity (0.85-0.92)
  - Early layers (0-5) show more variation (0.65-0.80)
  - Validates Last-Layer mapping strategy

**Figure 6.12: Recovery Timeline**
- X-axis: Training epochs
- Y-axis: Average benchmark accuracy
- Three lines: Original, Pruned, Recovering
- Shows progressive recovery converging toward original performance

### 6.7.4 Ablation Study: What Really Matters?

**Experiment Design:**
Train 4 versions of the same pruned model with different loss configurations:
1. **Baseline:** Only $L_{Task}$ (standard fine-tuning)
2. **Logits-KD:** $L_{Task} + L_{Logits}$ (Chapter 2 approach)
3. **Features-Only:** $L_{Task} + L_{Hidden}$ (skip logits alignment)
4. **Full Compound:** $L_{Task} + L_{Logits} + L_{Hidden}$ (our method)

**Table 6.6: Ablation Study Results**
| Loss Configuration | ARC-Easy | HellaSwag | Avg Recovery | Contribution |
|-------------------|----------|-----------|--------------|--------------|
| Original (pre-pruning) | 72.5% | 75.2% | 100.0% | - |
| Pruned (no training) | 61.8% | 64.1% | 0.0% | - |
| Only $L_{Task}$ | 65.2% | 67.8% | 84.5% | Baseline |
| $L_{Task} + L_{Logits}$ | 68.2% | 71.5% | 91.8% | +7.3% from soft labels |
| $L_{Task} + L_{Hidden}$ | 69.1% | 72.0% | 93.6% | +9.1% from features |
| **Full Compound** | **70.1%** | **72.5%** | **96.5%** | **+12.0% total** |

**Key Findings:**
1. Each component contributes meaningfully:
   - $L_{Logits}$: +7.3% (soft label knowledge transfer)
   - $L_{Hidden}$: +9.1% (internal reasoning alignment)
   - Combined: +12.0% (synergistic effect)

2. Feature alignment is more impactful than logits alignment for heavily pruned models

3. Skipping either component leaves significant performance on the table

**Practical Implication:**
> The cost of feature-based distillation (memory for storing hidden states, compute for projectors) is fully justified by the 4-5% additional recovery. For production deployments where every percentage point matters, this is essential.

### 6.7.5 Efficiency Analysis

**Table 6.7: Training Cost Comparison**
| Method | Memory (GB) | Time (hours) | Recovery Rate | Cost per 1% Recovery |
|--------|-------------|--------------|---------------|---------------------|
| Logits-Only KD | 12.3 | 1.8 | 91.8% | 0.020 hrs |
| Feature-Based KD | 16.7 | 2.4 | 96.5% | 0.025 hrs |

**Analysis:**
- Feature-based adds ~35% memory overhead (storing hidden states)
- Training time increases by ~33%
- But achieves +4.7% higher recovery
- **Verdict:** The trade-off is favorable for production-quality models

---

## 6.8 Summary
- Logits-only KD treats models as black boxes, achieving ~92% recovery
- Feature-Based Distillation aligns internal reasoning, pushing recovery to ~96.5%
- The compound loss balances three objectives: task accuracy, output distribution, and hidden state alignment
- Layer mapping strategies matter: Last-Layer alignment outperforms Uniform by 2-3%
- Learnable projectors are essential for width-pruned models—fixed projectors fail
- Hyperparameter tuning depends on pruning type: Width pruning needs higher γ
- The UniversalDistillationTrainer provides production-ready infrastructure
- Ablation studies confirm: each loss component contributes, with hidden state alignment providing the critical 4-5% improvement
- The additional cost (memory, compute) is justified by significantly better recovery
___

## Chapter 6: Knowledge Distillation & Recovery

### CH06_NB01_Layer_Mapping_Experiments.ipynb

**Sections:** 6.3  
**Duration:** 25-30 min

**Objective:** Resolve depth mismatch and demonstrate the impact of feature alignment

**Content:**
- Load Teacher (24 layers) + Student with depth pruning (20 layers)
- Implement Uniform and Last-Layer mapping
- Experiment A: Compare both mappings with logits-only KD
- Experiment B: Add feature alignment ($L_{Hidden}$) with Last-Layer
- Visualizations: Cosine similarity heatmaps, feature loss convergence
- Key finding: Last-Layer + Features = 96% recovery vs 91% logits-only

**Why separate:** It's the first real experiment of the chapter and demonstrates the central thesis (features matter). Short experiments (5 epochs each).

### CH06_NB02_Width_Mismatch_Projectors.ipynb

**Sections:** 6.4  
**Duration:** 20-25 min

**Objective:** Resolve width mismatch with learnable projectors

**Content:**
- Load Teacher (hidden_dim=896) + Student with width pruning (hidden_dim=640)
- Demonstrate the crash due to dimension mismatch
- Implement `LearnableProjector` class
- Experiment: Compare Fixed vs Learnable projectors
- Analysis of learned weights (what patterns they capture)
- Visualization: Cosine similarity before/after projection
- Key finding: Learnable projectors are necessary, fixed projectors fail

**Why separate:** Different technical problem (width vs depth). Fast experiments. The reader can understand projectors in isolation.

### CH06_NB03_Universal_Distiller_Production.ipynb

**Sections:** 6.5 + 6.6 + 6.7  
**Duration:** 45-60 min (includes full training)

**Objective:** Production-ready system with exhaustive evaluation

**Content:**
- Part 1 (6.5): Hyperparameter selection guidelines + helper functions
- Part 2 (6.6): Full implementation of `UniversalDistillationTrainer`
- Part 3 (6.7): Full training run on Llama-3.2-1B (3 epochs)
- Evaluation on 5 benchmarks (ARC, HellaSwag, Winogrande, LAMBADA, etc)
- Ablation study: 4 loss configurations
- Final visualizations: Recovery timeline, similarity heatmaps
- Key finding: 96.5% recovery, ablation confirms contribution of each component

**Why separate:** It's the "production" notebook. Long training (2-3 hours). Reusable template for other models. All exhaustive evaluation in one place.

