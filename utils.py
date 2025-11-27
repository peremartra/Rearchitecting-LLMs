try:
    import lm_eval
    import transformers
    import optipfair
    import torch
    import json
    import gc
    import langdetect
    from tqdm import tqdm
    import numpy as np 
except ImportError as e:
    raise ImportError(
        f"Missing required library: {e.name}\n"
        "Install all dependencies with:\n"
        "  pip install optipfair lm-eval transformers torch langdetect"
    )

def clear_gpu_cache():
    """Clear GPU cache completely"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def measure_detailed_performance(model, tokenizer, data_source, num_runs=3, max_new_tokens=50, max_samples=None):
    """
    Measures inference performance metrics with scientific rigor.
    
    Adapted to use robust timing and warmup logic while maintaining 
    the original interface for DataLoaders.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        data_source: DataLoader to sample from (expects batch['input_ids'])
        num_runs: Number of runs per sample for averaging (Latency focus)
        max_new_tokens: Tokens to generate per sample
        max_samples: Limit number of samples (None = all available)

    Returns:
        dict with timing statistics:
            - avg_latency_sec: Mean end-to-end latency across all measurements 
              (num_samples × num_runs total measurements)
            - std_latency_sec: Standard deviation of latency
            - avg_tokens_per_generation: Mean tokens generated per generation
            - throughput_tokens_per_sec: Overall throughput (total_tokens / total_time)
            - num_unique_samples: Number of unique input samples tested
            - num_runs_per_sample: Number of runs performed per sample
            - total_measurements: Total number of generation runs performed
            - total_tokens: Total tokens generated across all runs
    """
    device = model.device
    model.eval()
    
    # --- 1. DATA PREPARATION (Maintains original logic) ---
    samples = []
    # Flatten the data_source to get the list of input tensors
    for batch in data_source:
        current_batch_input_ids = batch['input_ids']
        for i in range(len(current_batch_input_ids)):
            samples.append(current_batch_input_ids[i])
            if max_samples and len(samples) >= max_samples:
                break
        if max_samples and len(samples) >= max_samples:
            break

    if max_samples:
        samples = samples[:max_samples]

    # Edge case: No samples available
    if not samples:
        print("⚠️  No samples to measure")
        return {
            'avg_latency_sec': 0.0,
            'std_latency_sec': 0.0,
            'avg_tokens_per_generation': 0.0,
            'throughput_tokens_per_sec': 0.0,
            'num_unique_samples': 0,
            'num_runs_per_sample': num_runs,
            'total_measurements': 0,
            'total_tokens': 0
        }

    print(f"Measuring performance on {len(samples)} samples ({num_runs} runs each)...")

    # --- 2. GPU WARM-UP ---
    # Critical to "warm up" the GPU to load kernels and allocators
    print("   🔥 Performing GPU Warm-up...")
    warmup_input = samples[0].unsqueeze(0).to(device)
    with torch.no_grad():
        # Perform 2 warmup passes (without measuring)
        for _ in range(2):
            model.generate(
                warmup_input,
                max_new_tokens=max_new_tokens,  # Use same length as test
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure warmup completed

    # --- 3. MEASUREMENT LOOP ---
    latencies = []
    total_tokens_generated = 0
    total_time_accumulated = 0

    with torch.no_grad():
        for sample in tqdm(samples, desc="Performance test"):
            input_ids = sample.unsqueeze(0).to(device)

            for _ in range(num_runs):
                # Synchronize before starting the clock (Vital for precision)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # Synchronize before stopping the clock
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()

                # Calculations
                elapsed = end_time - start_time
                num_new_tokens = outputs.shape[1] - input_ids.shape[1]

                # Store raw metrics
                latencies.append(elapsed)
                total_tokens_generated += num_new_tokens
                total_time_accumulated += elapsed

    # --- 4. METRICS CALCULATION (Robust logic) ---
    # Average Latency (End-to-End Latency)
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    # Average tokens per generation (FIX: removed redundant np.mean)
    avg_tokens_per_gen = total_tokens_generated / len(latencies) if latencies else 0.0

    # Tokens per Second (Global Throughput)
    # Calculated as Total Tokens / Total Time (More stable than averaging ratios)
    throughput = total_tokens_generated / total_time_accumulated if total_time_accumulated > 0 else 0.0

    # --- 5. RETURN WITH EXPLICIT TYPES ---
    return {
        'avg_latency_sec': float(avg_latency),
        'std_latency_sec': float(std_latency),
        'avg_tokens_per_generation': float(avg_tokens_per_gen),
        'throughput_tokens_per_sec': float(throughput),
        'num_unique_samples': int(len(samples)),
        'num_runs_per_sample': int(num_runs),
        'total_measurements': int(len(latencies)),
        'total_tokens': int(total_tokens_generated)
    }

def model_evaluation(model_obj, tokenizer, tasks, device='cuda', limit=None, batch_size=4):
    """
    Runs lm-eval on a PyTorch model object already in memory.

    Args:
        model_obj: The PyTorch model object to evaluate.
        tokenizer: The tokenizer object.
        tasks (list): A list of task names.
        limit (int): The number of samples per task.
    """
    print(f"Starting lm-eval on model '{model_obj.config._name_or_path}' for tasks: {tasks}")
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    
    # Wrap the local model object and tokenizer for lm-eval
    model_wrapper = HFLM(
        pretrained=model_obj,
        tokenizer=tokenizer,
        device=str(device)
    )
    
    # Parse tasks to handle both dict and string formats
    task_names = []
    task_fewshot_map = {}
    limit_str = f"(limit={limit})" if limit else "(full dataset)"
    for task in tasks:
        if isinstance(task, dict):
            task_name = task["name"]
            task_names.append(task_name)
            task_fewshot_map[task_name] = task["num_fewshot"]
        else:
            # Backward compatibility: simple string list
            task_names.append(task)
            task_fewshot_map[task] = 0

    print(f"\n{'='*70}")
    print(f"Tasks: {task_names} {limit_str}")
    print(f"Few-shot config: {task_fewshot_map}")
    print(f"{'='*70}\n")
    

    fewshot_value = list(task_fewshot_map.values())[0]
    results = evaluator.simple_evaluate(
        model=model_wrapper,
        tasks=task_names,
        num_fewshot=fewshot_value,
        limit=limit,
        device=str(device),
        batch_size=batch_size, 
    )

    # Format results for clean display
    formatted_results = {}
    for task_name, res in results["results"].items():
        # Extract relevant metrics based on task type
        if 'perplexity,none' in res:
            # Perplexity tasks (wikitext, lambada)
            formatted_results[task_name] = {
                'perplexity': f"{res.get('perplexity,none', 0):.2f}",
                'word_perplexity': f"{res.get('word_perplexity,none', 0):.2f}",
                'bits_per_byte': f"{res.get('bits_per_byte,none', 0):.4f}", 
                'accuracy': f"{res.get('acc,none', 0):.4f}",
            }
        elif 'acc,none' in res:
            # Accuracy tasks (boolq, arc, hellaswag, etc.)
            formatted_results[task_name] = {
                'accuracy': f"{res.get('acc,none', 0):.4f}",
                'acc_norm': f"{res.get('acc_norm,none', 0):.4f}" if 'acc_norm,none' in res else "N/A"
            }
        else:
            # Fallback: store all numeric metrics
            formatted_results[task_name] = {
                k: f"{v:.4f}" for k, v in res.items() 
                if isinstance(v, (int, float))
            }
    return formatted_results

def evaluate_metrics(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Create labels, ignoring padding (-100 = ignore_index)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            # Forward pass
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Only real tokens (no padding)
            num_real_tokens = attention_mask.sum().item()

            total_loss += outputs.loss.item() * num_real_tokens
            total_tokens += num_real_tokens

    # metrics
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return {
        'loss': avg_loss,
        'perplexity': perplexity
    }

def generate_text(model, tokenizer, prompt: str, device='cuda', max_new_tokens: int = 50) -> str:
    """Generate text with the model"""
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
