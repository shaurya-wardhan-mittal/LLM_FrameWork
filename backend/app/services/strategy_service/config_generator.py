"""Auto-generate training hyperparameters."""

from typing import Any


def estimate_vram_gb(params_b: float, strategy: str, max_seq_length: int) -> float:
    seq_factor = max_seq_length / 2048
    if strategy == "full":
        return params_b * 2.2 * seq_factor
    if strategy == "lora":
        return params_b * 1.2 * seq_factor
    return params_b * 0.75 * seq_factor  # qlora


def select_strategy(free_vram_gb: float, params_b: float, max_seq_length: int) -> tuple[str, str]:
    full_need = estimate_vram_gb(params_b, "full", max_seq_length)
    lora_need = estimate_vram_gb(params_b, "lora", max_seq_length)
    qlora_need = estimate_vram_gb(params_b, "qlora", max_seq_length)

    if free_vram_gb >= full_need * 1.1:
        return "full", f"VRAM {free_vram_gb:.1f}GB ≥ full FT estimate {full_need:.1f}GB"
    if free_vram_gb >= lora_need * 1.1:
        return "lora", f"VRAM {free_vram_gb:.1f}GB ≥ LoRA estimate {lora_need:.1f}GB"
    return "qlora", f"VRAM {free_vram_gb:.1f}GB; QLoRA estimate {qlora_need:.1f}GB"


def generate_training_config(
    *,
    strategy: str,
    row_count: int,
    params_b: float,
    max_seq_length: int = 2048,
) -> dict[str, Any]:
    # Scale epochs inversely with dataset size
    if row_count < 500:
        epochs = 5
    elif row_count < 5000:
        epochs = 3
    else:
        epochs = 2

    # Learning rate scales slightly with model size
    if params_b >= 13:
        lr = 1e-4
    elif params_b >= 7:
        lr = 1.5e-4
    else:
        lr = 2e-4

    # Batch size from strategy
    if strategy == "full":
        batch = 1
        grad_accum = 8
        lora_r = 0
        lora_alpha = 0
        load_4bit = False
    elif strategy == "lora":
        batch = 2
        grad_accum = 4
        lora_r = 32 if params_b >= 7 else 16
        lora_alpha = lora_r
        load_4bit = False
    else:
        batch = 2
        grad_accum = 4
        lora_r = 16
        lora_alpha = 16
        load_4bit = True

    return {
        "learning_rate": lr,
        "per_device_train_batch_size": batch,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": epochs,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "warmup_steps": min(50, max(5, row_count // 100)),
        "save_steps": 100,
        "logging_steps": 10,
        "eval_steps": 100,
        "load_in_4bit": load_4bit,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
    }
