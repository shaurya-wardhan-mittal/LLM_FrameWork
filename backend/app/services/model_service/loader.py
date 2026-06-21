"""Unsloth model loading for the local training workflow."""

try:
    import unsloth
except ImportError:
    pass

from typing import Any, Tuple


def load_model_and_tokenizer(
    model_id: str,
    *,
    max_seq_length: int = 2048,
    strategy: str = "qlora",
    hf_token: str | None = None,
) -> Tuple[Any, Any]:
    """
    Load model via Unsloth FastLanguageModel.
    Deferred import so API container does not require CUDA.
    """
    from unsloth import FastLanguageModel

    load_in_4bit = (strategy == "qlora") or ("4bit" in model_id.lower())
    full_finetuning = strategy == "full" and not load_in_4bit

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        full_finetuning=full_finetuning,
        
    )
    return model, tokenizer


def apply_lora_adapters(
    model,
    *,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
) -> Any:
    from unsloth import FastLanguageModel

    if hasattr(model, "peft_config") and model.peft_config:
        return model

    return FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
