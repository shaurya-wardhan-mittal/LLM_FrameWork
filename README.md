# Automatic Fine-Tune Framework

Upload one dataset and the framework handles the workflow:

1. Detect JSON, JSONL, CSV, or ShareGPT format.
2. Validate, clean, and normalize the records.
3. Inspect the local GPU and choose Full FT, LoRA, or QLoRA.
4. Generate training settings.
5. Train and evaluate with Unsloth.
6. Save the adapter under `data/runs/<run-id>/exports/adapter`.

There is no PostgreSQL, Redis, Celery, authentication, or multi-step job setup. Run
state is stored as JSON next to each local training run.

## Start

```powershell
./scripts/start-local.ps1
```

Open [http://localhost:3003](http://localhost:3003), then drop in a dataset.

The lightweight API dependencies are installed automatically. Before performing
real training, install the GPU dependencies once:

```powershell
backend\.venv\Scripts\pip.exe install -r backend\requirements-worker.txt
```

## Configuration

Copy `.env.example` to `.env` only when you need to change the default model,
sequence length, Hugging Face token, or frontend API URL.

## API

The UI uses only two workflow routes:

```text
POST /upload
GET  /runs/{run_id}
```

`POST /upload` prepares the dataset immediately and starts training in the local
API process. The status route reports the current stage and final adapter path.
