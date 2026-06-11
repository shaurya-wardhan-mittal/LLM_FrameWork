"""HuggingFace Trainer callbacks for metrics and logging."""

import logging
from typing import Any

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class JobEventCallback(TrainerCallback):
    def __init__(self, job_id: str, on_metric=None):
        self.job_id = job_id
        self.on_metric = on_metric

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        message = f"step={state.global_step} " + " ".join(f"{k}={v}" for k, v in logs.items())
        logger.info("%s: %s", self.job_id, message)
        if "loss" in logs and self.on_metric:
            self.on_metric(
                step=state.global_step,
                epoch=int(state.epoch or 0),
                train_loss=logs.get("loss"),
                eval_loss=logs.get("eval_loss"),
                learning_rate=logs.get("learning_rate"),
            )

    def on_save(self, args, state, control, **kwargs):
        logger.info("%s: checkpoint step=%s path=%s", self.job_id, state.global_step, args.output_dir)
