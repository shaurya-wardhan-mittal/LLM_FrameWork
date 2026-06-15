from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.services.model_service.service import ModelService
from app.workflow import create_run, get_run, train_run

settings = get_settings()

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def prepare_storage() -> None:
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
