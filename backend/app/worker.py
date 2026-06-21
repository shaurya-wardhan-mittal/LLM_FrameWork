import sys
from pathlib import Path
import os
import signal
import sys
from pathlib import Path

# Add backend root directory to sys.path so app module can be found
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def run_worker(run_id: str):
    print("PID:", os.getpid(), flush=True)
    print("Parent PID:", os.getppid(), flush=True)
    print("Python:", sys.executable, flush=True)


    print("Python:", sys.executable)
    print("WORKER STARTED", flush=True)

    print("About to import unsloth", flush=True)

    def handler(sig, frame):
        print(f"Received signal: {sig}", flush=True)

    signal.signal(signal.SIGINT, handler)

    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handler)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handler)
        
    import unsloth
    print("Imported unsloth", flush=True)

    print("About to import config", flush=True)
    from app.config import get_settings
    print("Imported config", flush=True)

    print("About to import workflow", flush=True)
    from app.workflow import get_run, _update_state, _run_dir
    print("Imported workflow", flush=True)

    print("About to import TrainingPipeline", flush=True)
    from app.services.trainer_service.pipeline import TrainingPipeline
    print("Imported TrainingPipeline", flush=True)

    settings = get_settings()
    state = get_run(run_id)
    if not state:
        print(f"Error: Run {run_id} not found.", file=sys.stderr)
        return

    try:
        print("Creating pipeline...", flush=True)
        pipeline = TrainingPipeline(
            run_id,
            on_status=lambda stage: _update_state(
                run_id,
                status="running" if stage != "completed" else "completed",
                stage=stage.replace("_", " ").title(),
            ),
        )
        print("Pipeline created", flush=True)

        print("Calling pipeline.run()", flush=True)

        result = pipeline.run(
            base_model_id=state["model_id"],
            strategy=state["strategy"],
            training_config=state["training_config"],
            cleaned_path=str(_run_dir(run_id) / "dataset.jsonl"),
            hf_token=settings.hf_token,
        )

        print("pipeline.run() returned", flush=True)
        print("Result:", result, flush=True)

        _update_state(
            run_id,
            status="completed",
            stage="Completed",
            output_path=result["export_path"],
            result=result,
        )
        print("Training successfully completed.", flush=True)

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"Training failed: {exc}\n{tb}", file=sys.stderr)
        _update_state(
            run_id,
            status="failed",
            stage="Failed",
            error=f"{exc}\n{tb}",
        )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.worker <run_id>", file=sys.stderr)
        sys.exit(1)
    run_worker(sys.argv[1])
