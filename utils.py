#try:
import lm_eval
import transformers
import optipfair
import torch
import json
import gc
import langdetect
import time
from tqdm import tqdm
import numpy as np 
import codecarbon
#except ImportError as e:
#    raise ImportError(
#        f"Missing required library: {e.name}\n"
#        "Install all dependencies with:\n"
#        "  pip install optipfair lm-eval transformers torch langdetect"
#    )

def clear_gpu_cache():
    """Clear GPU cache completely"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def measure_detailed_performance(model, tokenizer, data_source, num_runs=3, max_new_tokens=50, max_samples=None):
    """
    Measures inference performance.
    
    OPTIMIZED VERSION: Supports Batching and Mixed Precision (AMP) internally 
    while maintaining the exact same interface and return format.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        data_source: DataLoader to sample from (expects batch['input_ids'])
        num_runs: Number of runs per batch for averaging (Latency focus)
        max_new_tokens: Tokens to generate per sample
        max_samples: Limit number of samples (None = all available)

    Returns:
        dict with timing statistics (Compatible with original format):
            - avg_latency_sec: Mean latency per generation call (per batch if batched)
            - std_latency_sec: Standard deviation of latency
            - avg_tokens_per_generation: Mean tokens generated per call
            - throughput_tokens_per_sec: Overall throughput (total_tokens / total_time)
            - num_unique_samples: Number of unique input samples tested
            - num_runs_per_sample: Number of runs performed per batch
            - total_measurements: Total number of generation runs performed
            - total_tokens: Total tokens generated across all runs
            - (NEW) avg_latency_per_sample_sec: Normalized latency per single sample
            - (NEW) dtype_used: Precision format used
    """
    device = model.device
    model.eval()

    # --- OPTIMIZATION SETUP ---
    # Auto-detect best available precision (BF16 for Ampere+, FP16 otherwise)
    use_amp = torch.cuda.is_available()
    dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"🚀 Optimization active: Using Mixed Precision ({dtype}) if available.")
    
    # --- 1. DATA PREPARATION (Optimized) ---
    # We store BATCHES now, not flattened samples, to allow parallel GPU usage.
    batches = []
    total_samples_count = 0
    
    for batch in data_source:
        # Check current batch size
        current_bs = batch['input_ids'].shape[0]
        
        # Stop if we exceed max_samples
        if max_samples and total_samples_count >= max_samples:
            break
            
        batches.append(batch)
        total_samples_count += current_bs

    # Edge case: No samples available
    if not batches:
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

    print(f"Measuring performance on {len(batches)} batches (Total samples: {total_samples_count}, {num_runs} runs each)...")

    # --- 2. GPU WARM-UP ---
    # Critical to "warm up" the GPU to load kernels and allocators
    print("   🔥 Performing GPU Warm-up...")
    warmup_batch = batches[0]
    warmup_input = warmup_batch['input_ids'].to(device)
    # Handle attention mask if present (critical for batches)
    warmup_attn = warmup_batch['attention_mask'].to(device) if 'attention_mask' in warmup_batch else None

    with torch.no_grad():
        # Use Autocast for warmup too
        with torch.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            for _ in range(2):
                model.generate(
                    warmup_input,
                    attention_mask=warmup_attn,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure warmup completed

    # --- 3. MEASUREMENT LOOP ---
    latencies = [] # Stores latency per generation call (batch latency)
    total_tokens_generated = 0
    total_time_accumulated = 0

    with torch.no_grad():
        for batch in tqdm(batches, desc="Performance test"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
            
            # Dimensions for calculation
            batch_size = input_ids.shape[0]
            input_length = input_ids.shape[1]

            for _ in range(num_runs):
                # Synchronize before starting the clock (Vital for precision)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                # OPTIMIZATION: Mixed Precision Context
                with torch.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
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
                # Calculate total new tokens: (Total Len - Input Len) * Batch Size
                num_new_tokens = (outputs.shape[1] - input_length) * batch_size

                # Store raw metrics
                latencies.append(elapsed)
                total_tokens_generated += num_new_tokens
                total_time_accumulated += elapsed

    # --- 4. METRICS CALCULATION (Robust logic) ---
    # Average Latency (Per generation call)
    # NOTE: If batch_size > 1, this is the latency of the BATCH, not single sample.
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    # NEW: Average Latency Normalized per sample (for fair comparison)
    # We estimate this by dividing batch latency by average batch size approx
    avg_batch_size = total_samples_count / len(batches) if batches else 1
    avg_latency_per_sample = avg_latency / avg_batch_size

    # Average tokens per generation call
    avg_tokens_per_gen = total_tokens_generated / len(latencies) if latencies else 0.0

    # Tokens per Second (Global Throughput) - The gold standard for GPU saturation
    throughput = total_tokens_generated / total_time_accumulated if total_time_accumulated > 0 else 0.0

    # --- 5. RETURN WITH EXPLICIT TYPES ---
    return {
        'avg_latency_sec': float(avg_latency),        # Latency of the .generate() call
        'std_latency_sec': float(std_latency),
        'avg_tokens_per_generation': float(avg_tokens_per_gen),
        'throughput_tokens_per_sec': float(throughput),
        'num_unique_samples': int(total_samples_count), # Updated to reflect real count
        'num_runs_per_sample': int(num_runs),           # Actually runs per batch now
        'total_measurements': int(len(latencies)),
        'total_tokens': int(total_tokens_generated),
        # New compatible metrics:
        'avg_latency_per_sample_sec': float(avg_latency_per_sample),
        'dtype_used': str(dtype)
    }

def model_evaluation(model_obj, tokenizer, tasks, device='cuda', limit=None, batch_size=4):
    """
    Runs evaluation tasks on a loaded PyTorch model using the lm-evaluation-harness.
    
    This function wraps a pre-loaded model and tokenizer into an HFLM wrapper, 
    parses task configurations (supporting both simple strings and few-shot dicts), 
    and executes the evaluation. Tasks with different few-shot settings are 
    automatically grouped and evaluated separately. Results are post-processed 
    to return only the most relevant metrics (perplexity, accuracy, etc.).
    
    Args:
        model_obj (PreTrainedModel): The Hugging Face/PyTorch model object.
        tokenizer (PreTrainedTokenizer): The associated tokenizer.
        tasks (list[str | dict]): A list of tasks. Can be task name strings or 
            dicts with keys 'name' (str) and 'num_fewshot' (int).
        device (str): Device to run evaluation on (e.g., 'cuda', 'cpu'). 
            Defaults to 'cuda'.
        limit (int, optional): Number of samples per task for quick testing. 
            If None, the full dataset is used.
        batch_size (int): Batch size for the evaluator. Defaults to 4.
    
    Returns:
        dict: A cleaned results dictionary where keys are task names and values 
            are nested dicts containing relevant metrics like 'accuracy', 
            'perplexity', or 'acc_norm'.
    
    Example:
        >>> tasks = [{"name": "hellaswag", "num_fewshot": 5}, "wikitext"]
        >>> results = model_evaluation(model, tokenizer, tasks)
    """
    print(f"Starting lm-eval on model '{model_obj.config._name_or_path}' for tasks: {tasks}")
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from collections import defaultdict
    
    # Wrap the local model object and tokenizer for lm-eval
    model_wrapper = HFLM(
        pretrained=model_obj,
        tokenizer=tokenizer,
        device=str(device)
    )
    
    # Parse tasks and group by num_fewshot for efficient evaluation
    fewshot_groups = defaultdict(list)
    task_fewshot_map = {}
    
    for task in tasks:
        if isinstance(task, dict):
            task_name = task["name"]
            num_fewshot = task.get("num_fewshot", 0)
            fewshot_groups[num_fewshot].append(task_name)
            task_fewshot_map[task_name] = num_fewshot
        else:
            # Backward compatibility: simple string list defaults to 0-shot
            fewshot_groups[0].append(task)
            task_fewshot_map[task] = 0
    
    limit_str = f"(limit={limit})" if limit else "(full dataset)"
    print(f"\n{'='*70}")
    print(f"Tasks grouped by few-shot: {dict(fewshot_groups)} {limit_str}")
    print(f"Task-level few-shot config: {task_fewshot_map}")
    print(f"{'='*70}\n")
    
    # Run evaluation for each few-shot group
    all_results = {}
    for num_fewshot, task_list in fewshot_groups.items():
        print(f"Evaluating {len(task_list)} task(s) with {num_fewshot}-shot learning...")
        results = evaluator.simple_evaluate(
            model=model_wrapper,
            tasks=task_list,
            num_fewshot=num_fewshot,
            limit=limit,
            device=str(device),
            batch_size=batch_size,
        )
        all_results.update(results["results"])
    
    # Define priority metrics with their formatting
    PRIORITY_METRICS = {
        'perplexity': (['perplexity,none', 'perplexity'], ':.2f'),
        'word_perplexity': (['word_perplexity,none', 'word_perplexity'], ':.2f'),
        'bits_per_byte': (['bits_per_byte,none', 'bits_per_byte'], ':.4f'),
        'accuracy': (['acc,none', 'acc'], ':.4f'),
        'acc_norm': (['acc_norm,none', 'acc_norm'], ':.4f'),
        'f1': (['f1,none', 'f1'], ':.4f'),
        'exact_match': (['exact_match,none', 'em'], ':.4f'),
    }
    
    # Format results for clean display
    formatted_results = {}
    for task_name, res in all_results.items():
        formatted_results[task_name] = {}
        
        # Extract priority metrics with specific formatting
        for metric_name, (possible_keys, fmt) in PRIORITY_METRICS.items():
            for key in possible_keys:
                if key in res:
                    val = res[key]
                    # Apply dynamic formatting using format() builtin
                    formatted_results[task_name][metric_name] = format(val, fmt.strip(':'))
                    break
        
        # If no priority metrics found, fallback to all numeric metrics
        if not formatted_results[task_name]:
            formatted_results[task_name] = {
                k: f"{v:.4f}" for k, v in res.items() 
                if isinstance(v, (int, float))
            }
    
    return formatted_results
    
def evaluate_metrics(model, dataloader, device='cuda'):
    """
    Evaluates a language model by calculating average loss and perplexity.

    The function sets the model to evaluation mode, iterates through the 
    provided dataloader, and computes metrics based on real tokens, 
    effectively ignoring padding in the loss calculation.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The validation or test data loader.
        device (str, optional): The device to run the evaluation on (e.g., 'cuda', 'cpu'). 
            Defaults to 'cuda'.

    Returns:
        dict: A dictionary containing the evaluation results:
            - 'loss' (float): The average cross-entropy loss per token.
            - 'perplexity' (float): The model's perplexity score.
    """
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
    
def calibrate_idle_power(device="cuda", duration_seconds=30, verbose=True):
    """
    Measure GPU idle power consumption to establish baseline.
    
    This should be run ONCE at the start of the notebook, before any model loading.
    
    Args:
        device (str): Device to calibrate ("cuda" or "cpu")
        duration_seconds (int): How long to measure (30s recommended)
        verbose (bool): Print progress messages
    
    Returns:
        dict: {
            "idle_power_watts": float,
            "idle_energy_kwh": float,
            "measurement_duration_s": float,
            "gpu_temp_celsius": float,
            "gpu_name": str,
            "timestamp": str
        }
    """
    import torch
    import time
    from codecarbon import EmissionsTracker
    from datetime import datetime
    
    if verbose:
        print(f"🔋 Starting idle power calibration ({duration_seconds}s)...")
        print(f"   Clearing GPU cache...")
    
    # Clear GPU to true idle state
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Initialize tracker
    tracker = EmissionsTracker(
        project_name="idle_calibration",
        measure_power_secs=1,  # Sample every second
        save_to_file=False,
        log_level="error"  # Suppress warnings
    )
    
    # Measure idle consumption
    tracker.start()
    start_time = time.time()
    
    # Just wait (GPU should be completely idle)
    if verbose:
        print(f"   Measuring idle power for {duration_seconds}s...")
    time.sleep(duration_seconds)
    
    emissions_kwh = tracker.stop()
    actual_duration = time.time() - start_time
    
    # Calculate average power
    idle_power_watts = (emissions_kwh * 1000) / (actual_duration / 3600)  # kWh -> W
    
    # Capture GPU state
    gpu_info = {}
    if device == "cuda" and torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_temp_celsius": torch.cuda.temperature() if hasattr(torch.cuda, 'temperature') else None,
            "gpu_power_limit_w": torch.cuda.get_device_properties(0).total_memory / (1024**3) * 10  # Rough estimate
        }
    
    calibration_result = {
        "idle_power_watts": idle_power_watts,
        "idle_energy_kwh": emissions_kwh,
        "measurement_duration_s": actual_duration,
        "timestamp": datetime.now().isoformat(),
        **gpu_info
    }
    
    if verbose:
        print(f"✅ Calibration complete!")
        print(f"   Idle Power: {idle_power_watts:.2f} W")
        print(f"   Idle Energy (30s): {emissions_kwh:.6f} kWh")
        if gpu_info.get("gpu_temp_celsius"):
            print(f"   GPU Temperature: {gpu_info['gpu_temp_celsius']:.1f}°C")
    
    return calibration_result

def measure_energy_consumption(model, tokenizer, data_source, idle_power_watts=None, num_runs=1, max_new_tokens=50, max_samples=None):
    """
    Measures the net energy consumption of a model removing idle power.
    
    Mirrors the structure of 'measure_detailed_performance' for consistency
    in evaluation pipelines.

    Args:
        model: PyTorch model to evaluate.
        tokenizer: Tokenizer.
        data_source: DataLoader to sample from (expects batch['input_ids']).
        idle_power_watts: Idle GPU power in Watts for net correction.
            - If None: auto-calibrates (adds ~30s)
            - If 0: no idle correction applied
            - If float > 0: uses provided value
        num_runs: Number of runs per sample (typically 1 for energy measurement).
        max_new_tokens: Tokens to generate per sample.
        max_samples: Limit number of samples (None = all available).

    Returns:
        dict: Energy metrics (kWh, Joules, Joules/Token, CO2).
            Note: CO2 is calculated on raw energy (before idle correction)
            as CodeCarbon doesn't support post-hoc adjustments.
    """
    import time
    import torch
    from tqdm import tqdm
    from codecarbon import EmissionsTracker
    
    device = model.device
    model.eval()

    # --- 1. DATA PREPARATION (Identical to performance function) ---
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
        print("⚠️ No samples to measure.")
        return {
            "duration_sec": 0.0,
            "total_tokens": 0,
            "num_unique_samples": 0,
            "num_runs_per_sample": num_runs,
            "total_measurements": 0,
            "energy_raw_kwh": 0.0,
            "energy_idle_correction_kwh": 0.0,
            "energy_net_kwh": 0.0,
            "efficiency_joules_per_token": 0.0,
            "co2_emissions_kg": 0.0
        }

    # --- 2. AUTO-CALIBRATION (Only if idle_power_watts is None) ---
    if idle_power_watts is None:
        print("   ⚙️ Auto-calibrating idle power (30s)...")
        calibration_result = calibrate_idle_power(
            device=device, 
            duration_seconds=30,
            verbose=False
        )
        idle_power_watts = calibration_result["idle_power_watts"]
        print(f"   ✓ Measured idle power: {idle_power_watts:.2f}W")

    print(f"🌍 Measuring energy on {len(samples)} samples ({num_runs} runs each)...")

    # --- 3. GPU WARM-UP (Identical to performance function) ---
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

    # --- 4. INITIALIZE ENERGY TRACKER ---
    # save_to_file=False keeps the environment clean for the book reader
    tracker = EmissionsTracker(
        project_name="manning_energy_test",
        measure_power_secs=1,
        save_to_file=False,
        log_level="error"
    )

    # --- 5. MEASUREMENT LOOP (Structure mirrors performance function) ---
    total_tokens_generated = 0
    
    # Start tracking before the measurement loop
    tracker.start()
    start_time = time.time()

    try:
        with torch.no_grad():
            for sample in tqdm(samples, desc="Energy measurement"):
                input_ids = sample.unsqueeze(0).to(device)

                for _ in range(num_runs):
                    # Synchronize before starting generation (Vital for precision)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    
                    # Synchronize after generation
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # Count only NEW tokens for efficiency calculation
                    num_new_tokens = outputs.shape[1] - input_ids.shape[1]
                    total_tokens_generated += num_new_tokens

    finally:
        # Stop tracker immediately after inference
        emissions_raw_kwh = tracker.stop()

    duration_seconds = time.time() - start_time

    # --- 6. NET ENERGY CALCULATION ---
    # We subtract the energy the GPU would have consumed at idle
    # Formula: Net = Total - (Idle_Watts * Time_Seconds) / 3_600_000
    
    idle_energy_kwh = (idle_power_watts * duration_seconds) / 3_600_000
    energy_net_kwh = max(0.0, emissions_raw_kwh - idle_energy_kwh)

    # --- 7. EFFICIENCY METRICS ---
    # Joules are better than kWh for small comparisons (1 kWh = 3.6M Joules)
    total_joules_net = energy_net_kwh * 3_600_000
    joules_per_token = total_joules_net / total_tokens_generated if total_tokens_generated > 0 else 0.0

    # --- 8. CO2 EMISSIONS ---
    # Note: CodeCarbon calculates this on raw energy consumption
    # For net CO2, you would need: co2_net = co2_raw * (energy_net / energy_raw)
    co2_raw_kg = float(tracker.final_emissions)
    
    # --- 9. RETURN WITH EXPLICIT TYPES (Consistent with performance function) ---
    return {
        "duration_sec": float(duration_seconds),
        "total_tokens": int(total_tokens_generated),
        "num_unique_samples": int(len(samples)),
        "num_runs_per_sample": int(num_runs),
        "total_measurements": int(len(samples) * num_runs),
        "energy_raw_kwh": float(emissions_raw_kwh),
        "energy_idle_correction_kwh": float(idle_energy_kwh),
        "energy_net_kwh": float(energy_net_kwh),
        "efficiency_joules_per_token": float(joules_per_token),
        "co2_emissions_kg": co2_raw_kg
    }
