import shutil
import sys
from pathlib import Path
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import get_settings
from app.services.model_service.service import ModelService
from app.workflow import create_run, get_run, train_run, _update_state

settings = get_settings()

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def prepare_storage() -> None:
    print("MAIN PYTHON:", sys.executable, flush=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.get("/models")
def list_models() -> dict:
    service = ModelService()
    return {"models": service.list_catalog()}


@app.post("/upload", status_code=status.HTTP_202_ACCEPTED)

async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_id: str | None = Form(None),
) -> dict:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded dataset is empty.")

    if model_id is not None:
        service = ModelService()
        available_models = {m["id"] for m in service.list_catalog()}
        if model_id not in available_models:
            raise HTTPException(status_code=400, detail="Selected model is not available in the catalog.")

    try:
        run = create_run(file.filename or "dataset.jsonl", content, model_id=model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    background_tasks.add_task(train_run, run["id"])
    return run


@app.get("/runs/{run_id}")
def run_status(run_id: str) -> dict:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


# Additional trigger endpoints (aliases for /upload)
@app.post("/train", status_code=status.HTTP_202_ACCEPTED)
async def train_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_id: str | None = Form(None),
) -> dict:
    return await upload_dataset(background_tasks, file, model_id)


@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_id: str | None = Form(None),
) -> dict:
    return await upload_dataset(background_tasks, file, model_id)


@app.post("/finetune", status_code=status.HTTP_202_ACCEPTED)
async def finetune_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_id: str | None = Form(None),
) -> dict:
    return await upload_dataset(background_tasks, file, model_id)


# Training start/restart endpoints for existing runs
def _trigger_existing_run(run_id: str, background_tasks: BackgroundTasks) -> dict:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    
    if run.get("status") == "running":
        raise HTTPException(status_code=400, detail="Run is already running.")
        
    _update_state(run_id, status="queued", error=None)
    background_tasks.add_task(train_run, run_id)
    return {"status": "queued", "run_id": run_id}


@app.post("/runs/{run_id}/train", status_code=status.HTTP_202_ACCEPTED)
def run_train_endpoint(run_id: str, background_tasks: BackgroundTasks) -> dict:
    return _trigger_existing_run(run_id, background_tasks)


@app.post("/runs/{run_id}/start", status_code=status.HTTP_202_ACCEPTED)
def run_start_endpoint(run_id: str, background_tasks: BackgroundTasks) -> dict:
    return _trigger_existing_run(run_id, background_tasks)


# Endpoints to retrieve/download the final fine-tuned model
def get_or_create_model_zip(run_id: str) -> Path:
    run_dir = settings.runs_dir / run_id
    adapter_dir = run_dir / "exports" / "adapter"
    if not adapter_dir.exists():
        raise HTTPException(status_code=404, detail="Final fine-tuned model export not found.")
        
    zip_path = run_dir / "exports" / "adapter"
    target_zip = run_dir / "exports" / "adapter.zip"
    
    if not target_zip.exists():
        try:
            shutil.make_archive(str(zip_path), 'zip', root_dir=str(adapter_dir))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to package model files: {exc}")
            
    return target_zip


@app.get("/runs/{run_id}/model")
def download_model(run_id: str):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    if run.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Run is not completed. Current status: {run.get('status')}")
        
    target_zip = get_or_create_model_zip(run_id)
    return FileResponse(
        path=target_zip,
        filename=f"model-{run_id}.zip",
        media_type="application/zip",
    )


@app.get("/runs/{run_id}/model/files")
def list_model_files(run_id: str) -> dict:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    run_dir = settings.runs_dir / run_id
    adapter_dir = run_dir / "exports" / "adapter"
    if not adapter_dir.exists():
        raise HTTPException(status_code=404, detail="Model files have not been exported yet.")
        
    files = []
    for file_path in adapter_dir.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(adapter_dir)
            stat = file_path.stat()
            files.append({
                "name": str(rel_path),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
            
    return {"run_id": run_id, "files": files}


@app.get("/runs/{run_id}/model/files/{file_name:path}")
def download_model_file(run_id: str, file_name: str):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    run_dir = settings.runs_dir / run_id
    adapter_dir = run_dir / "exports" / "adapter"
    file_path = (adapter_dir / file_name).resolve()
    
    # Prevent directory traversal attacks
    try:
        file_path.relative_to(adapter_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied.")
        
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
        
    return FileResponse(
        path=file_path,
        filename=file_path.name,
    )


@app.get("/runs/{run_id}/logs")
def get_run_logs(run_id: str) -> dict:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    run_dir = settings.runs_dir / run_id
    log_file = run_dir / "training.log"
    
    logs = ""
    if log_file.exists():
        try:
            logs = log_file.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logs = f"Error reading log file: {exc}"
            
    return {"run_id": run_id, "logs": logs}

