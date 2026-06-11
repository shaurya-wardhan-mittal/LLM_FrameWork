"""Dataset validation and cleaning."""

from typing import Any

REQUIRED_HINTS = {
    "instruction": {"instruction", "input", "output", "prompt", "response"},
    "chat": {"messages", "conversations"},
    "qa": {"question", "answer", "context"},
    "code": {"instruction", "code", "output", "solution"},
}


def detect_dataset_type(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "instruction"
    keys = set()
    for row in rows[:50]:
        keys.update(k.lower() for k in row.keys())
    if keys & {"conversations", "messages"}:
        return "chat"
    if keys & {"question", "answer"} and "instruction" not in keys:
        return "qa"
    if keys & {"code", "solution"} or any("```" in str(v) for v in rows[0].values()):
        return "code"
    return "instruction"


def validate_rows(rows: list[dict[str, Any]], dataset_type: str) -> tuple[list[dict], list[str]]:
    issues: list[str] = []
    cleaned: list[dict] = []
    hints = REQUIRED_HINTS.get(dataset_type, REQUIRED_HINTS["instruction"])

    for i, row in enumerate(rows):
        if not row:
            issues.append(f"Row {i}: empty record")
            continue
        row_keys = {k.lower() for k in row.keys()}
        if not row_keys & hints:
            issues.append(f"Row {i}: missing expected fields for {dataset_type}")
        # Strip whitespace from strings
        cleaned_row = {
            k: v.strip() if isinstance(v, str) else v
            for k, v in row.items()
            if v is not None and (not isinstance(v, str) or v.strip())
        }
        if cleaned_row:
            cleaned.append(cleaned_row)
        else:
            issues.append(f"Row {i}: all fields empty after cleaning")

    return cleaned, issues


def build_quality_report(
    rows: list[dict],
    cleaned: list[dict],
    issues: list[str],
    dataset_type: str,
) -> dict[str, Any]:
    lengths = [sum(len(str(v)) for v in r.values()) for r in cleaned[:500]]
    return {
        "dataset_type": dataset_type,
        "total_rows": len(rows),
        "valid_rows": len(cleaned),
        "dropped_rows": len(rows) - len(cleaned),
        "issue_count": len(issues),
        "issues_sample": issues[:20],
        "avg_char_length": sum(lengths) / len(lengths) if lengths else 0,
        "max_char_length": max(lengths) if lengths else 0,
        "field_coverage": _field_coverage(cleaned),
    }


def _field_coverage(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    all_keys: set[str] = set()
    for r in rows[:100]:
        all_keys.update(r.keys())
    coverage = {}
    for key in all_keys:
        coverage[key] = sum(1 for r in rows if key in r and r[key]) / len(rows)
    return coverage
