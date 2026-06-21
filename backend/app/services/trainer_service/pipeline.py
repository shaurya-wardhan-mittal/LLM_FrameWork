"""Full training orchestration pipeline using Unsloth + TRL."""

try:
    import unsloth
except ImportError:
    pass

try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    pass

import json
import logging
import random
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.config import get_settings
from app.services.evaluation_service.service import EvaluationService
from app.services.export_service.service import ExportService
from app.services.model_service.loader import apply_lora_adapters, load_model_and_tokenizer
from app.services.strategy_service.gpu_probe import snapshot_resources
from app.services.trainer_service.callbacks import JobEventCallback

logger = logging.getLogger(__name__)
settings = get_settings()


class TrainingPipeline:
    """Runs a complete local fine-tuning workflow."""

    def __init__(
        self,
        job_id: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_metric: Callable[..., None] | None = None,
        on_checkpoint: Callable[..., None] | None = None,
        on_resource: Callable[[dict], None] | None = None,
    ):
        self.job_id = job_id
        self.workspace = settings.runs_dir / job_id
        self.on_status = on_status
        self.on_metric = on_metric
        self.on_checkpoint = on_checkpoint
        self.on_resource = on_resource
        self._stop_monitor = threading.Event()

    def _set_status(self, status: str) -> None:
        if self.on_status:
            self.on_status(status)

    def _start_resource_monitor(self) -> threading.Thread:
        def _loop():
            while not self._stop_monitor.is_set():
                snap = snapshot_resources()
                if self.on_resource:
                    self.on_resource(snap)
                time.sleep(5)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        return t

    def prepare_dataset(self, cleaned_path: str, val_ratio: float = 0.05) -> tuple[Path, Path]:
        self._set_status("preparing")
        dataset_dir = self.workspace / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        lines = Path(cleaned_path).read_text(encoding="utf-8").strip().split("\n")
        random.shuffle(lines)
        split_idx = max(1, int(len(lines) * (1 - val_ratio)))
        train_lines = lines[:split_idx]
        eval_lines = lines[split_idx:] or lines[-max(1, len(lines) // 20):]

        train_path = dataset_dir / "train.jsonl"
        eval_path = dataset_dir / "eval.jsonl"
        train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
        eval_path.write_text("\n".join(eval_lines) + "\n", encoding="utf-8")
        return train_path, eval_path

    def run(
        self,
        *,
        base_model_id: str,
        strategy: str,
        training_config: dict[str, Any],
        cleaned_path: str,
        checkpoint_path: str | None = None,
        hf_token: str | None = None,
    ) -> dict[str, Any]:
        print("PIPELINE RUN STARTED", flush=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        config_path = self.workspace / "config.json"
        config_path.write_text(json.dumps(training_config, indent=2), encoding="utf-8")

        monitor = self._start_resource_monitor()
        try:
            train_path, eval_path = self.prepare_dataset(cleaned_path)

            self._set_status("training")
            logger.info("Loading model %s", base_model_id)

            max_seq = training_config.get("max_seq_length", 2048)

            # If the model is 4bit quantized, full fine-tuning is impossible.
            # We override the strategy to qlora for safety (e.g. if the run was queued before the code fix).
            if "4bit" in base_model_id.lower() and strategy.lower() == "full":
                logger.warning("Forcing strategy to 'qlora' because the base model is quantized.")
                strategy = "qlora"
                if training_config.get("lora_r", 0) == 0:
                    training_config["lora_r"] = 16
                if training_config.get("lora_alpha", 0) == 0:
                    training_config["lora_alpha"] = 16

            model, tokenizer = load_model_and_tokenizer(
                base_model_id,
                max_seq_length=max_seq,
                strategy=strategy,
                hf_token=hf_token 
            )
            
            print("Strategy:", repr(strategy), flush=True)
            print("Model loaded:", type(model))
            print("Before LoRA:", hasattr(model, "peft_config"), flush=True)

            strategy = strategy.lower()

            if strategy in ("lora", "qlora"):
                try:
                    print("Applying LoRA adapters...")

                    model = apply_lora_adapters(
                        model,
                        lora_r=training_config.get("lora_r", 16),
                        lora_alpha=training_config.get("lora_alpha", 16),
                        lora_dropout=training_config.get("lora_dropout", 0.0),
                    )
                    print("LoRA attached successfully")
                    print("After LoRA:", hasattr(model, "peft_config"), flush=True)
                    print("PEFT Config:", getattr(model, "peft_config", None))

                    trainable = sum(
                        p.numel()
                        for p in model.parameters()
                        if p.requires_grad
                    )
                    total = sum(p.numel() for p in model.parameters()if p.requires_grad)

                    print(f"Trainable params: {trainable:,}", flush=True)
                    print(f"Total params: {total:,}")

                except Exception as e:
                    import traceback
                    print("FAILED TO ATTACH LORA")
                    print("Exception:", repr(e))
                    print(traceback.format_exc())
                    raise

               

            from datasets import load_dataset
            from unsloth import is_bfloat16_supported

            train_ds = load_dataset("json", data_files=str(train_path), split="train")
            eval_ds = load_dataset("json", data_files=str(eval_path), split="train")

            output_dir = str(self.workspace / "checkpoints")
            args = SFTConfig(
                per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
                gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
                warmup_steps=training_config.get("warmup_steps", 10),
                num_train_epochs=training_config.get("num_train_epochs", 3),
                learning_rate=training_config.get("learning_rate", 2e-4),
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=training_config.get("logging_steps", 10),
                optim=training_config.get("optim", "adamw_8bit"),
                weight_decay=training_config.get("weight_decay", 0.01),
                lr_scheduler_type=training_config.get("lr_scheduler_type", "linear"),
                seed=3407,
                output_dir=output_dir,
                save_steps=training_config.get("save_steps", 100),
                eval_strategy="steps",
                eval_steps=training_config.get("eval_steps", 100),
                report_to="none",
                max_seq_length=max_seq,
                dataset_text_field="text",
            )

            callback = JobEventCallback(self.job_id, on_metric=self.on_metric)

            print("\n===== BEFORE SFTTRAINER =====", flush=True)
            print("Model type:", type(model))
            print("Has peft:", hasattr(model, "peft_config"))
            print("PEFT config:", getattr(model, "peft_config", None))

            trainable = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )
            print("Trainable params:", trainable)
            print("============================\n", flush=True)

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                args=args,
                callbacks=[callback],
            )

            resume = checkpoint_path if checkpoint_path and Path(checkpoint_path).exists() else None
            trainer.train(resume_from_checkpoint=resume)

            if self.on_checkpoint:
                self.on_checkpoint(step=trainer.state.global_step, path=output_dir)

            self._set_status("evaluating")
            eval_service = EvaluationService()
            eval_result = eval_service.run_eval(trainer, eval_ds)

            self._set_status("exporting")
            export_dir = self.workspace / "exports" / "adapter"
            export_dir.mkdir(parents=True, exist_ok=True)
            ExportService.save_lora(model, tokenizer, export_dir)

            self._set_status("completed")
            return {
                "checkpoint_path": output_dir,
                "export_path": str(export_dir),
                "eval": eval_result,
                "final_step": trainer.state.global_step,
            }
        except Exception as exc:
            self._set_status("failed")
            logger.exception("Training run %s failed: %s", self.job_id, exc)
            raise
        finally:
            self._stop_monitor.set()
            monitor.join(timeout=2)
