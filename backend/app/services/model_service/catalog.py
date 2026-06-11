"""Curated Unsloth-compatible model catalog."""

CURATED_MODELS = [
    {
        "id": "unsloth/Llama-3.2-3B-bnb-4bit",
        "name": "Llama 3.2 3B (4-bit)",
        "params_b": 3.0,
        "context_length": 8192,
        "family": "llama",
    },
    {
        "id": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "name": "Llama 3.1 8B (4-bit)",
        "params_b": 8.0,
        "context_length": 8192,
        "family": "llama",
    },
    {
        "id": "unsloth/Qwen2.5-7B-bnb-4bit",
        "name": "Qwen 2.5 7B (4-bit)",
        "params_b": 7.0,
        "context_length": 32768,
        "family": "qwen",
    },
    {
        "id": "unsloth/gemma-2-9b-bnb-4bit",
        "name": "Gemma 2 9B (4-bit)",
        "params_b": 9.0,
        "context_length": 8192,
        "family": "gemma",
    },
    {
        "id": "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        "name": "Phi-3.5 Mini (4-bit)",
        "params_b": 3.8,
        "context_length": 4096,
        "family": "phi",
    },
]


def estimate_params_from_id(model_id: str) -> float:
    """Rough parameter count in billions from model id string."""
    lower = model_id.lower()
    for token in ("3b", "7b", "8b", "9b", "13b", "70b", "1b", "1.5b", "2b", "4b"):
        if token in lower:
            return float(token.replace("b", ""))
    for m in CURATED_MODELS:
        if m["id"] == model_id:
            return m["params_b"]
    return 7.0  # conservative default
