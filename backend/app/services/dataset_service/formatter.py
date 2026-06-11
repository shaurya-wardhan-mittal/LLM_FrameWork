"""Normalize rows to Unsloth SFT `text` field."""

import json
from typing import Any


def format_for_training(row: dict[str, Any], dataset_type: str) -> dict[str, str]:
    if dataset_type == "instruction":
        instruction = row.get("instruction") or row.get("prompt") or ""
        input_text = row.get("input") or ""
        output = row.get("output") or row.get("response") or ""
        if input_text:
            text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    elif dataset_type == "chat":
        text = _format_chat(row)
    elif dataset_type == "qa":
        q = row.get("question", "")
        a = row.get("answer", "")
        ctx = row.get("context", "")
        if ctx:
            text = f"### Context:\n{ctx}\n\n### Question:\n{q}\n\n### Answer:\n{a}"
        else:
            text = f"### Question:\n{q}\n\n### Answer:\n{a}"
    elif dataset_type == "code":
        instruction = row.get("instruction") or row.get("prompt") or ""
        code = row.get("code") or row.get("output") or row.get("solution") or ""
        text = f"### Instruction:\n{instruction}\n\n### Code:\n```\n{code}\n```"
    else:
        text = json.dumps(row, ensure_ascii=False)
    return {"text": text.strip()}


def _format_chat(row: dict[str, Any]) -> str:
    if "messages" in row:
        parts = []
        for msg in row["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)
    if "conversations" in row:
        parts = []
        for turn in row["conversations"]:
            role = turn.get("from", turn.get("role", "human"))
            value = turn.get("value", turn.get("content", ""))
            parts.append(f"<|{role}|>\n{value}")
        return "\n".join(parts)
    return str(row)


def write_jsonl(rows: list[dict[str, str]], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
