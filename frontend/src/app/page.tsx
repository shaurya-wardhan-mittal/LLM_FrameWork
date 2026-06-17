  "use client";

  import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  type Model = {
    id: string;
    name: string;
    params_b: number;
    context_length: number;
    family: string;
  };

  type Run = {
    id: string;
    status: "queued" | "running" | "completed" | "failed";
    stage: string;
    filename: string;
    format: string;
    dataset_type: string;
    row_count: number;
    model_id: string;
    strategy: string;
    strategy_rationale: string;
    estimated_vram_gb: number;
    output_path: string | null;
    error: string | null;
    quality_report: {
      valid_rows: number;
      dropped_rows: number;
      issue_count: number;
    };
  };

  export default function HomePage() {
    const [models, setModels] = useState<Model[]>([]);
    const [selectedModelId, setSelectedModelId] = useState<string>("");
    const [run, setRun] = useState<Run | null>(null);
    const [uploading, setUploading] = useState(false);
    const [dragging, setDragging] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
      // Fetch available models
      async function loadModels() {
        try {
          const response = await fetch(`${API_URL}/models`);
          if (response.ok) {
            const data = await response.json();
            setModels(data.models);
            // Set first model as default
            if (data.models.length > 0) {
              setSelectedModelId(data.models[0].id);
            }
          }
        } catch {
          console.error("Failed to load models");
        }
      }
      loadModels();
    }, []);

    useEffect(() => {
      if (!run || run.status === "completed" || run.status === "failed") return;

      const timer = window.setInterval(async () => {
        try {
          const response = await fetch(`${API_URL}/runs/${run.id}`, { cache: "no-store" });
          if (response.ok) setRun(await response.json());
        } catch {
          // The local API may be restarting; the next poll will retry.
        }
      }, 2000);

      return () => window.clearInterval(timer);
    }, [run]);

    async function upload(file: File) {
      setUploading(true);
      setError(null);
      setRun(null);
      const form = new FormData();
      form.append("file", file);
      form.append("model_id", selectedModelId);

      try {
        const response = await fetch(`${API_URL}/upload`, { method: "POST", body: form });
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || "Upload failed.");
        setRun(body);
      } catch (uploadError) {
        setError(uploadError instanceof Error ? uploadError.message : "Upload failed.");
      } finally {
        setUploading(false);
      }
    }

    function onFileChange(event: ChangeEvent<HTMLInputElement>) {
      const file = event.target.files?.[0];
      if (file) upload(file);
      event.target.value = "";
    }

    function onDrop(event: DragEvent<HTMLDivElement>) {
      event.preventDefault();
      setDragging(false);
      const file = event.dataTransfer.files?.[0];
      if (file) upload(file);
    }

    return (
      <main className="mx-auto flex min-h-screen max-w-5xl flex-col justify-center px-5 py-16">
        <div className="mb-10 max-w-3xl">
          <p className="mb-3 text-sm font-semibold uppercase tracking-[0.24em] text-sky-400">
            Automatic fine-tuning
          </p>
          <h1 className="text-4xl font-bold tracking-tight sm:text-6xl">
            Upload the dataset.
            <span className="block text-slate-400">The framework handles the rest.</span>
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-400">
            Format detection, cleaning, validation, GPU-aware strategy selection, training,
            evaluation, and adapter export run automatically.
          </p>
        </div>

        <div className="mb-6 max-w-3xl">
          <label className="block text-sm font-medium text-slate-300">Base Model</label>
          <select
            value={selectedModelId}
            onChange={(e) => setSelectedModelId(e.target.value)}
            disabled={uploading || models.length === 0}
            className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-slate-100 disabled:cursor-wait disabled:opacity-60"
          >
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name} ({model.params_b}B)
              </option>
            ))}
          </select>
          {models.length === 0 && (
            <p className="mt-1 text-xs text-slate-500">Loading models...</p>
          )}
        </div>

        <div
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`rounded-3xl border-2 border-dashed p-10 text-center transition sm:p-16 ${
            dragging ? "border-sky-400 bg-sky-400/10" : "border-slate-700 bg-slate-900/70"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".json,.jsonl,.csv"
            className="hidden"
            onChange={onFileChange}
          />
          <p className="text-xl font-semibold">
            {uploading ? "Preparing your dataset..." : "Drop a JSON, JSONL, CSV, or ShareGPT file here"}
          </p>
          <p className="mt-2 text-sm text-slate-500">No database, account, or setup workflow required.</p>
          <button
            type="button"
            disabled={uploading}
            onClick={() => inputRef.current?.click()}
            className="mt-7 rounded-xl bg-sky-500 px-6 py-3 font-semibold text-slate-950 transition hover:bg-sky-400 disabled:cursor-wait disabled:opacity-60"
          >
            {uploading ? "Uploading..." : "Choose dataset"}
          </button>
        </div>

        {error && (
          <div className="mt-6 rounded-2xl border border-red-900 bg-red-950/50 p-5 text-red-300">
            {error}
          </div>
        )}
        {run && <RunStatus run={run} />}
      </main>
    );
  }

  function RunStatus({ run }: { run: Run }) {
    const isActive = run.status === "queued" || run.status === "running";

    return (
      <section className="mt-8 rounded-3xl border border-slate-800 bg-slate-900 p-6 sm:p-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm text-slate-500">{run.filename}</p>
            <h2 className="mt-1 text-2xl font-semibold">{run.stage}</h2>
          </div>
          <span className={`rounded-full px-3 py-1 text-sm font-medium ${statusStyle(run.status)}`}>
            {isActive ? "Working" : run.status}
          </span>
        </div>

        {isActive && (
          <div className="mt-6 h-1.5 overflow-hidden rounded-full bg-slate-800">
            <div className="progress h-full w-1/3 rounded-full bg-sky-400" />
          </div>
        )}

        <div className="mt-7 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="Usable rows" value={String(run.row_count)} />
          <Stat label="Detected type" value={run.dataset_type} />
          <Stat label="Strategy" value={run.strategy.toUpperCase()} />
          <Stat label="VRAM estimate" value={`${run.estimated_vram_gb.toFixed(1)} GB`} />
        </div>

        <div className="mt-6 rounded-2xl bg-slate-950/70 p-5 text-sm leading-6 text-slate-400">
          <p><span className="text-slate-200">Model:</span> {run.model_id}</p>
          <p><span className="text-slate-200">Decision:</span> {run.strategy_rationale}</p>
          <p>
            <span className="text-slate-200">Validation:</span>{" "}
            {run.quality_report.valid_rows} valid, {run.quality_report.dropped_rows} dropped,{" "}
            {run.quality_report.issue_count} issues
          </p>
        </div>

        {run.output_path && (
          <p className="mt-6 rounded-2xl border border-emerald-900 bg-emerald-950/40 p-5 text-emerald-300">
            Adapter saved to <code>{run.output_path}</code>
          </p>
        )}
        {run.error && (
          <p className="mt-6 rounded-2xl border border-red-900 bg-red-950/50 p-5 text-red-300">
            {run.error}
          </p>
        )}
      </section>
    );
  }

  function Stat({ label, value }: { label: string; value: string }) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
        <p className="text-xs uppercase tracking-wider text-slate-500">{label}</p>
        <p className="mt-2 truncate font-semibold text-slate-100">{value}</p>
      </div>
    );
  }

  function statusStyle(status: Run["status"]) {
    if (status === "completed") return "bg-emerald-400/15 text-emerald-300";
    if (status === "failed") return "bg-red-400/15 text-red-300";
    return "bg-sky-400/15 text-sky-300";
  }
