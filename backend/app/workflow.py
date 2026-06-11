import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services.dataset_service.formatter import format_for_training, write_jsonl
from app.services.dataset_service.parsers import detect_format, parse_file
from app.services.dataset_service.validator import (
    build_quality_report,
    detect_dataset_type,
    validate_rows,
)
from app.services.model_service.catalog import estimate_params_from_id
from app.services.strategy_service.config_generator import (
    estimate_vram_gb,
    generate_training_config,
    select_strategy,
)
from app.services.strategy_service.gpu_probe import probe_gpu

settings = get_settings()
_state_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_dir(run_id: str) -> Path:
    return settings.runs_dir / run_id


def _state_path(run_id: str) -> Path:
    return _run_dir(run_id) / "status.json"


def _write_state(run_id: str, state: dict[str, Any]) -> dict[str, Any]:
    state["updated_at"] = _now()
    path = _state_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with _state_lock:
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(path)
    return state


def _update_state(run_id: str, **changes: Any) -> dict[str, Any]:
    state = get_run(run_id)
    if not state:
        raise FileNotFoundError(f"Run {run_id} does not exist")
    state.update(changes)
    return _write_state(run_id, state)


def get_run(run_id: str) -> dict[str, Any] | None:
    path = _state_path(run_id)
    if not path.exists():
        return None
    with _state_lock:
        return json.loads(path.read_text(encoding="utf-8"))


def create_run(filename: str, file_bytes: bytes) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix or ".jsonl"
    raw_path = run_dir / f"source{suffix}"
    raw_path.write_bytes(file_bytes)

    try:
        file_format = detect_format(safe_name, file_bytes[:8192])
        rows = parse_file(raw_path, file_format)
        if not rows:
            raise ValueError("The dataset does not contain any records.")

        dataset_type = detect_dataset_type(rows)
        cleaned_rows, issues = validate_rows(rows, dataset_type)
        if not cleaned_rows:
            raise ValueError("No usable records remained after validation.")

        quality_report = build_quality_report(rows, cleaned_rows, issues, dataset_type)
        cleaned_path = run_dir / "dataset.jsonl"
        write_jsonl(
            [format_for_training(row, dataset_type) for row in cleaned_rows],
            cleaned_path,
        )
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read this dataset: {exc}") from exc

    params_b = estimate_params_from_id(settings.default_model_id)
    gpu = probe_gpu()
    free_vram_gb = gpu.free_vram_mb / 1024 if gpu.available else 4.0
    strategy, rationale = select_strategy(
        free_vram_gb,
        params_b,
        settings.max_seq_length,
    )
    training_config = generate_training_config(
        strategy=strategy,
        row_count=len(cleaned_rows),
        params_b=params_b,
        max_seq_length=settings.max_seq_length,
    )

    state = {
        "id": run_id,
        "status": "queued",
        "stage": "Dataset prepared",
        "filename": safe_name,
        "format": file_format,
        "dataset_type": dataset_type,
        "row_count": len(cleaned_rows),
        "quality_report": quality_report,
        "model_id": settings.default_model_id,
        "strategy": strategy,
        "strategy_rationale": rationale,
        "estimated_vram_gb": estimate_vram_gb(
            params_b,
            strategy,
            settings.max_seq_length,
        ),
        "training_config": training_config,
        "output_path": None,
        "error": None,
        "created_at": _now(),
    }
    return _write_state(run_id, state)


def train_run(run_id: str) -> None:
    state = get_run(run_id)
    if not state:
        return

    try:
        _update_state(run_id, status="running", stage="Starting training")

        from app.services.trainer_service.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(
            run_id,
            on_status=lambda stage: _update_state(
                run_id,
                status="running" if stage != "completed" else "completed",
                stage=stage.replace("_", " ").title(),
            ),
        )
        result = pipeline.run(
            base_model_id=state["model_id"],
            strategy=state["strategy"],
            training_config=state["training_config"],
            cleaned_path=str(_run_dir(run_id) / "dataset.jsonl"),
            hf_token=settings.hf_token,
        )
        _update_state(
            run_id,
            status="completed",
            stage="Completed",
            output_path=result["export_path"],
            result=result,
        )
    except Exception as exc:
        _update_state(
            run_id,
            status="failed",
            stage="Failed",
            error=str(exc),
        )
