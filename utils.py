try:
    import lm_eval
    import transformers
    import optipfair
    import torch
    import json
    import gc
    
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

def model_evaluation(model_obj, tokenizer, tasks, device='cuda', limit=None):
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
    )

    # Format results for clean display
    formatted_results = {}
    for task_name, res in results["results"].items():
        # Look for accuracy ('acc') first, then perplexity ('ppl')
        if 'acc,none' in res:
            metric_val = res.get('acc,none', 0)
        elif 'ppl,none' in res:
             metric_val = res.get('ppl,none', 0)
        else:
            metric_val = 0 # Fallback

        formatted_results[task_name] = f"{metric_val:.4f}"

    print(json.dumps(formatted_results, indent=2))
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
