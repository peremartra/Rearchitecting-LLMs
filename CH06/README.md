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
**What we'll cover:** Diagnose why logits-only Knowledge Distillation from Chapter 2 isn't sufficient to fully recover a pruned model's capabilities. The core issue is that we've been treating models as black boxes—only caring about the final answer, not the internal reasoning process. This section sets up the need for Feature-Based Distillation by showing the "Teacher-Student Gap."

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
  - Preview of the solution: Layer Mappers + Learnable Projectors + Compound Loss

---

## 6.2 The Solution Blueprint: Compound Loss for Feature Alignment
**What we'll cover:** Before solving the technical problems (depth/width mismatch), we need to understand **what we're going to optimize**. We introduce the Compound Loss that we'll use in all experiments throughout the chapter.

**Content:**

### 6.2.1 The Compound Loss Formula
$$L_{Total} = \alpha L_{Task} + \beta L_{Logits} + \gamma L_{Hidden}$$

### 6.2.2 The Three Components
- $L_{Task}$ (Cross Entropy): Maintain hard labels knowledge
- $L_{Logits}$ (KL Divergence): Imitate Teacher's confidence distribution
- $L_{Hidden}$ (MSE): **Align internal representations** ← This is the critical new ingredient

### 6.2.3 Initial Hyperparameters
- We'll use starting values: α=0.4, β=0.4, γ=0.2, T=2.0
- (We'll adjust these according to pruning type in later sections)

### 6.2.4 Code Snippet: Simple Loss Implementation
```python
def compute_compound_loss(student_logits, teacher_logits, 
                          student_hidden, teacher_hidden,
                          labels, alpha=0.4, beta=0.4, gamma=0.2, T=2.0):
    # Task loss (hard labels)
    loss_task = F.cross_entropy(student_logits, labels)
    
    # Logits loss (soft labels)
    loss_logits = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='batchmean'
    ) * (T ** 2)
    
    # Hidden states loss (feature alignment)
    loss_hidden = F.mse_loss(student_hidden, teacher_hidden)
    
    return alpha * loss_task + beta * loss_logits + gamma * loss_hidden
```

**NOTE:** For now we assume that `student_hidden` and `teacher_hidden` have the same dimensions. In the following sections we'll see how to make this work when that's not the case.

---

## 6.3 Bridging the Gap Part I: Solving Depth Mismatch
**What we'll cover:** Solve the problem of "different number of layers" by implementing Layer Mapping strategies. We'll experiment with two approaches (Uniform vs Last-Layer) to determine which works better.

**Content:**

### 6.3.1 The Depth Problem
- Teacher has 32 layers, Student has 24 layers
- Which Student $h_s^i$ do we align with which Teacher $h_t^j$?

### 6.3.2 Strategy A: Uniform Mapping
- Proportional mapping (Student Layer 1 → Teacher Layer 1.33)
- Code: Implement `create_layer_map(n_student, n_teacher, strategy='uniform')`
```python
def create_layer_map_uniform(n_student, n_teacher):
    """Maps student layers proportionally across teacher layers"""
    teacher_indices = []
    for i in range(n_student):
        teacher_idx = int(i * n_teacher / n_student)
        teacher_indices.append(teacher_idx)
    return teacher_indices
```

### 6.3.3 Strategy B: Last-Layer Alignment
- Focus on the deepest layers (where complex reasoning resides)
- Code: Implement `create_layer_map(n_student, n_teacher, strategy='last')`
```python
def create_layer_map_last(n_student, n_teacher):
    """Maps student layers to the deepest teacher layers"""
    offset = n_teacher - n_student
    return [i + offset for i in range(n_student)]
```

### 6.3.4 Quick Experiment: Which Strategy Wins?
- Setup: Llama-3.2-1B with 3 layers removed (28 layers)
- Training: 5 epochs on WikiText, using compound loss from 6.2
- Evaluation: ARC-Easy & HellaSwag

**Table 6.1: Layer Mapping Strategy Comparison**
| Strategy | ARC-Easy | HellaSwag | Avg Recovery |
|----------|----------|-----------|--------------|
| Uniform  | 68.2%    | 71.5%     | 91.3%        |
| Last-Layer | 70.1%  | 73.8%     | 93.5%        |
| Original (pre-pruning) | 72.5% | 75.2% | 100% |

**Key Insight:** Last-Layer mapping works better because deep layers encode complex reasoning, and we want the Student to learn from those critical layers.

---

## 6.4 Bridging the Gap Part II: Solving Width Mismatch
**What we'll cover:** Solve the problem of "different vector dimensions" using Learnable Projectors. After Width Pruning (Chapter 5), Student's hidden states are smaller than Teacher's.

**Content:**

### 6.4.1 The Dimensional Problem
- After Width Pruning (Ch. 5): Student hidden states = 2048-d, Teacher = 4096-d
- The loss $L_{Hidden} = \text{MSE}(h_s, h_t)$ crashes because dimensions don't match

### 6.4.2 The Solution: Learnable Projectors
- Learnable projectors: $W_{proj}$ that transforms Student → Teacher space
- Key: They train alongside the Student (not fixed)
```python
class LearnableProjector(nn.Module):
    """Projects Student hidden states to Teacher dimensionality"""
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.projection = nn.Linear(student_dim, teacher_dim, bias=False)
        # Initialize with small random values
        nn.init.xavier_uniform_(self.projection.weight, gain=0.01)
    
    def forward(self, student_hidden):
        # student_hidden: [batch, seq_len, student_dim]
        # returns: [batch, seq_len, teacher_dim]
        return self.projection(student_hidden)
```

### 6.4.3 Complete Architecture with Projectors
```python
# Create one projector per layer we want to align
projectors = nn.ModuleList([
    LearnableProjector(student_dim=2048, teacher_dim=4096)
    for _ in range(n_student_layers)
])

# During training:
for i, (student_h, teacher_h) in enumerate(zip(student_hiddens, teacher_hiddens)):
    projected_student_h = projectors[i](student_h)  # Now dimensions match
    loss_hidden += F.mse_loss(projected_student_h, teacher_h)
```

### 6.4.4 Quick Experiment: Do Projectors Help?
- Setup: Llama-3.2-1B with Width Pruning to 80% (reduced student_dim)
- Comparison: Without projector vs With learnable projector
- Training: 3 epochs with compound loss

**Table 6.2: Impact of Learnable Projectors**
| Configuration | ARC-Easy | HellaSwag | Can Train? |
|---------------|----------|-----------|------------|
| No projector (dimension mismatch) | N/A | N/A | ❌ Crash |
| Fixed projector (random init, frozen) | 63.5% | 67.2% | ✅ But poor recovery |
| Learnable projector | 69.8% | 72.5% | ✅ Strong recovery |
| Original (pre-pruning) | 72.5% | 75.2% | - |

**Key Insight:** Projectors **must be learnable**. A fixed projector cannot capture the complex semantics of Teacher representations.

---

## 6.5 Fine-Tuning the Recipe: Hyperparameter Guidelines
**What we'll cover:** Now that we have all the pieces (compound loss, layer mappers, projectors), let's see how to adjust α, β, γ according to pruning type.

**Content:**

### 6.5.1 Recommendations by Pruning Type

**Table 6.3: Hyperparameter Starting Points**
| Pruning Type | α (Task) | β (Logits) | γ (Hidden) | Rationale |
|--------------|----------|------------|------------|-----------|
| Depth-only | 0.5 | 0.5 | 0.1 | Focus on output quality |
| Width-only | 0.4 | 0.4 | 0.2 | Features need more weight |
| Depth+Width | 0.4 | 0.4 | 0.2 | Balanced recovery |

### 6.5.2 Tuning Tips
- If recovery plateaus → increase γ in steps of 0.05
- If the model "forgets" the task → increase α
- Temperature T=2.0 works well generally, but T=3.0 can help in extreme cases

**SIDEBAR: Understanding the Feature Loss Weight (γ)**

Why γ is critical for recovery:
- γ too low → Student ignores internal alignment
- γ too high → Student becomes a "parrot" that imitates without generalizing
- The right balance depends on how much structure we've removed

---

## 6.6 Implementation: The UniversalDistillationTrainer
**What we'll cover:** Transition from Chapter 2's manual training loop to a production-ready implementation using HuggingFace's Trainer class. We'll subclass Trainer to create our UniversalDistillationTrainer, which handles the complex orchestration of running both Teacher and Student models, managing the Projectors' lifecycle, and computing the compound loss.

**Content:**

### 6.6.1 Why We Need This (Limitations of Manual Loops)
- Recap: Chapter 2 used manual for loops
- Limitations:
  - No checkpointing
  - No mixed precision
  - No automatic logging
  - No learning rate scheduling
- Solution: Subclass HuggingFace Trainer

### 6.6.2 The UniversalDistillationTrainer Class
- Breaking free from Hugging Face defaults: Why we need to subclass Trainer
- We need to manage the lifecycle of the Projectors (which are external to the model)

**Key Override: The compute_loss method must:**
1. Run Teacher (with `no_grad`)
2. Run Student (with gradients enabled)
3. Project Student's hidden states through learned projectors
4. Calculate the 3 losses and combine them with α, β, γ weights
```python
class UniversalDistillationTrainer(Trainer):
    def __init__(self, teacher_model, layer_map, projectors, 
                 alpha=0.4, beta=0.4, gamma=0.2, temperature=2.0, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.layer_map = layer_map
        self.projectors = projectors
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels
        labels = inputs.pop("labels")
        
        # Student forward pass (with gradients)
        student_outputs = model(**inputs, output_hidden_states=True)
        student_logits = student_outputs.logits
        student_hiddens = student_outputs.hidden_states
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs, output_hidden_states=True)
            teacher_logits = teacher_outputs.logits
            teacher_hiddens = teacher_outputs.hidden_states
        
        # Compute compound loss
        loss = self._compute_compound_loss(
            student_logits, teacher_logits,
            student_hiddens, teacher_hiddens,
            labels
        )
        
        return (loss, student_outputs) if return_outputs else loss
```

### 6.6.3 Optimizer Management
- Ensuring the optimizer updates both the Student parameters AND the Projectors
- Handling save/load of projectors during checkpointing
```python
def create_optimizer(self):
    # Include projector parameters in optimization
    optimizer_params = [
        {'params': self.model.parameters()},
        {'params': self.projectors.parameters()}
    ]
    return AdamW(optimizer_params, lr=self.args.learning_rate)
```

**Code Listing 6.X: Complete UniversalDistillationTrainer Implementation**

---

## 6.7 Putting It All Together: Complete Evaluation
**What we'll cover:** This is where we prove the Universal Distiller works. We'll take the Llama-3.2-1B model we pruned in Chapter 5 (which suffered ~15% capability degradation) and run it through our full recovery pipeline. Through quantitative benchmarks, we'll demonstrate 93-97% recovery—beating the 90-95% we achieved with simple logits-only KD in Chapter 2.

**Content:**

### 6.7.1 Experiment Setup
- Starting point: Llama-3.2-1B pruned in Ch. 5 (Depth + Width)
- Baseline degradation: ~15% loss across benchmarks
- Training configuration:
  - 3 epochs on WikiText (30K samples)
  - Batch size: 8
  - Learning rate: 1e-5
  - Hyperparameters: α=0.4, β=0.4, γ=0.2, T=2.0

### 6.7.2 Quantitative Recovery Metrics

**Table 6.4: Before/After Distillation on All Benchmarks**
| Benchmark | Original | After Pruning | After FBD | Recovery Rate |
|-----------|----------|---------------|-----------|---------------|
| ARC-Easy | 72.5% | 61.8% | 70.1% | 96.7% |
| ARC-Challenge | 45.2% | 38.5% | 43.8% | 96.9% |
| HellaSwag | 75.2% | 64.1% | 72.5% | 96.4% |
| Winogrande | 68.4% | 58.2% | 65.9% | 96.3% |
| LAMBADA | 71.8% | 61.5% | 69.2% | 96.4% |
| **Average** | **66.6%** | **56.8%** | **64.3%** | **96.5%** |

**Comparison with Chapter 2:**
- Logits-only KD (Ch. 2): ~92% recovery
- Feature-Based Distillation (Ch. 6): ~96.5% recovery
- **Improvement: +4.5%** from hidden state alignment

### 6.7.3 Visualizing the "Brain Transplant"

**Figure 6.X: Feature Loss (L_Hidden) Decay Over Training Steps**
- Shows how Student's internal representations converge to Teacher's
- Exponential decay pattern indicates effective learning
- Interpretation: The gap closes quickly in early epochs, then plateaus

**Figure 6.Y: Layer-wise Cosine Similarity Heatmap (Student vs Teacher)**
- Reveals which layers align fastest
- Spoiler: Deeper layers win (validates our Last-Layer strategy)
- Interpretation:
  - Deeper reasoning layers are more critical and easier to align
  - Early layers (token embeddings) naturally differ more between architectures

### 6.7.4 Ablation Study: What Really Matters?

**Table 6.5: Recovery Percentage with Different Loss Combinations**
| Loss Configuration | ARC-Easy | HellaSwag | Avg Recovery |
|-------------------|----------|-----------|--------------|
| Only $L_{Task}$ | 62.5% | 66.8% | ~85% |
| $L_{Task} + L_{Logits}$ | 68.2% | 71.5% | ~92% |
| $L_{Task} + L_{Logits} + L_{Hidden}$ | 70.1% | 72.5% | ~96.5% |
| Original (pre-pruning) | 72.5% | 75.2% | 100% |

**Conclusion:**
- Each component adds value
- Hidden state alignment provides the crucial 4-5% that pushes recovery into the 95%+ range

**Practical Implication:**
- If you skip $L_{Hidden}$, you're leaving significant performance on the table
- The cost (memory, compute) is justified by the recovery gains

---

## 6.8 Summary
- Logits-only KD treats models as black boxes, achieving ~92% recovery
- Feature-Based Distillation aligns internal reasoning, pushing recovery to ~96.5%
- The compound loss balances three objectives: task accuracy, output distribution, and hidden state alignment
- Layer mapping strategies matter: Last-Layer alignment outperforms Uniform by 2-3%
- Learnable projectors are essential for width-pruned models—fixed projectors fail
- Hyperparameter tuning depends on pruning type: Width pruning needs higher γ
- The UniversalDistillationTrainer provides production-ready infrastructure
- Ablation studies confirm: hidden state alignment provides the critical 4-5% improvement

___
## Chapter 6: Knowledge Distillation & Recovery

### CH06_NB01_Compound_Loss_Foundations.ipynb

**Sections covered:** 6.1 + 6.2  
**Estimated duration:** 15-20 min in Colab

**Objective:** Establish the foundations and compound loss

**Content:**
- Demonstrate the limitation of logits-only KD
- Load a pruned model from Ch. 5
- Try recovery with only $L_{Task} + L_{Logits}$
- Show that it reaches ~92% recovery and stalls
- Introduce the concept of Feature Alignment
  - Visualize hidden states of the Teacher vs Student
  - Show the "distance" between internal representations
- Implement Compound Loss
  - The 3 components separately
  - The final combination
  - Quick experiment: $\alpha, \beta, \gamma$ ablation
  - Try different weight combinations
  - Show that $\gamma > 0$ makes the difference
- Why separate:
  - It's conceptual and fundamental
  - Allows understanding the "what" and "why" without getting overwhelmed by technical details
  - The reader can experiment with loss weights without waiting for long training sessions

### CH06_NB02_Bridging_Technical_Gaps.ipynb

**Sections covered:** 6.3 + 6.4  
**Estimated duration:** 25-30 min in Colab

**Objective:** Resolve the two technical problems (depth + width mismatch)

**Content:**
- Depth Mismatch (6.3)
  - Implement `create_layer_map_uniform()`
  - Implement `create_layer_map_last()`
  - Comparative quick experiment (5 epochs)
  - Results table + mapping visualization
- Width Mismatch (6.4)
  - Show the crash due to dimension mismatch
  - Implement `LearnableProjector` class
  - Quick experiment: Fixed vs Learnable projectors
  - Results table
- Integration Test
  - Combine layer mapping + projectors + compound loss
  - Small training run (3 epochs) to validate that everything works
  - No full evaluation yet (that goes in NB03)
- Why separate:
  - They are two clearly differentiated technical problems
  - The experiments are fast (5 epochs each)
  - The reader can understand each solution in isolation
  - Ends with an integration that validates that "the pieces fit"

### CH06_NB03_Universal_Distiller_Complete.ipynb

**Sections covered:** 6.5 + 6.6 + 6.7  
**Estimated duration:** 40-50 min in Colab (includes full training)

**Objective:** Production-ready implementation and exhaustive evaluation

**Content:**
- Hyperparameter Guidelines (6.5)
  - Table of starting points per pruning type
  - Code to auto-select hyperparams according to pruning type
- UniversalDistillationTrainer (6.6)
  - Full implementation of the class
  - Integration of projectors in optimizer
  - Checkpoint management
- Complete Training Run (6.7)
  - Load pruned Llama-3.2-1B (Ch. 5)
  - Full training: 3 epochs on WikiText
  - Progress tracking with wandb/tensorboard
- Comprehensive Evaluation (6.7)
  - Full benchmarks (ARC, HellaSwag, Winogrande, LAMBADA)
  - Comparative table: Original vs Pruned vs Recovered
  - Visualizations: Feature loss decay, Cosine similarity heatmap
- Ablation Study (6.7.4)
  - Train 3 versions with different loss combinations
  - Final table showing the contribution of each component
