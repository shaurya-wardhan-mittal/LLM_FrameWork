"""Evaluation and training report generation."""

import json
from pathlib import Path
from typing import Any


class EvaluationService:
    def run_eval(self, trainer, eval_dataset) -> dict[str, Any]:
        metrics = trainer.evaluate()
        perplexity = None
        if "eval_loss" in metrics and metrics["eval_loss"] is not None:
            import math

            perplexity = math.exp(min(metrics["eval_loss"], 20))
        return {
            "eval_loss": metrics.get("eval_loss"),
            "perplexity": perplexity,
            "raw": metrics,
        }

    def build_report(
        self,
        *,
        job_id: str,
        training_config: dict,
        metrics: list[dict],
        eval_result: dict,
        output_dir: Path,
    ) -> dict[str, Any]:
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        loss_series = [
            {"step": m["step"], "train_loss": m.get("train_loss"), "eval_loss": m.get("eval_loss")}
            for m in metrics
            if m.get("train_loss") is not None
        ]
        curve_path = reports_dir / "loss_curve.json"
        curve_path.write_text(json.dumps(loss_series, indent=2), encoding="utf-8")

        try:
            self._plot_loss_curve(loss_series, reports_dir / "loss_curve.png")
            png_path = str(reports_dir / "loss_curve.png")
        except Exception:
            png_path = None

        summary = {
            "job_id": job_id,
            "training_config": training_config,
            "final_train_loss": loss_series[-1]["train_loss"] if loss_series else None,
            "eval": eval_result,
            "total_steps": loss_series[-1]["step"] if loss_series else 0,
        }
        return {"summary": summary, "loss_curve_path": png_path or str(curve_path)}

    def _plot_loss_curve(self, series: list[dict], path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [p["step"] for p in series]
        losses = [p["train_loss"] for p in series]
        plt.figure(figsize=(8, 4))
        plt.plot(steps, losses, label="train_loss")
        eval_points = [(p["step"], p["eval_loss"]) for p in series if p.get("eval_loss")]
        if eval_points:
            plt.plot([e[0] for e in eval_points], [e[1] for e in eval_points], label="eval_loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
