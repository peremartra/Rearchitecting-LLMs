try:
    import lm_eval
    import transformers
    import optipfair
except ImportError as e:
    raise ImportError(
        f"Missing required library: {e.name}\n"
        "Install all dependencies with:\n"
        "  pip install optipfair lm-eval transformers torch langdetect"
    )

def model_evaluation(model_obj, tokenizer, tasks, limit=None):
    """
    Runs lm-eval on a PyTorch model object already in memory.

    Args:
        model_obj: The PyTorch model object to evaluate.
        tokenizer: The tokenizer object.
        tasks (list): A list of task names.
        limit (int): The number of samples per task.
    """
    print(f"Starting lm-eval on model '{model_obj.config._name_or_path}' for tasks: {tasks}")

    # Wrap the local model object and tokenizer for lm-eval
    model_wrapper = HFLM(
        pretrained=model_obj,
        tokenizer=tokenizer,
        device=str(device)
    )

    results = evaluator.simple_evaluate(
        model=model_wrapper,
        tasks=tasks,
        num_fewshot=0,
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
