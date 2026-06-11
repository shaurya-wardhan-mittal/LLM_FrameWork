"""Parsers for JSON, CSV, JSONL, and ShareGPT formats."""

import csv
import json
from pathlib import Path
from typing import Any, Iterator


def detect_format(filename: str, content_sample: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".jsonl"):
        return "jsonl"
    if lower.endswith(".json"):
        try:
            data = json.loads(content_sample.decode("utf-8", errors="replace"))
            if isinstance(data, list) and data and "conversations" in data[0]:
                return "sharegpt"
        except json.JSONDecodeError:
            pass
        return "json"
    if lower.endswith(".json"):
        return "json"
    # Sniff content
    text = content_sample.decode("utf-8", errors="replace").strip()
    if text.startswith("["):
        try:
            data = json.loads(text[: min(len(text), 50000)])
            if isinstance(data, list) and data and "conversations" in data[0]:
                return "sharegpt"
        except json.JSONDecodeError:
            pass
        return "json"
    if "\n" in text and text.split("\n", 1)[0].strip().startswith("{"):
        return "jsonl"
    return "jsonl"


def parse_file(path: Path, fmt: str) -> list[dict[str, Any]]:
    if fmt == "csv":
        return _parse_csv(path)
    if fmt == "jsonl":
        return _parse_jsonl(path)
    if fmt == "sharegpt":
        return _parse_sharegpt(path)
    return _parse_json(path)


def _parse_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    return rows


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return [data]


def _parse_sharegpt(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data
