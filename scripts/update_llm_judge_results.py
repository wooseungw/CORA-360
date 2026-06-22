#!/usr/bin/env python3
"""Update EXPERIMENT_PLAN.md with the latest LLM-Judge scores.

Reads stats JSON files from runs/baseline/{model}/llm_judge/ and
rewrites the LLM-Judge column in the relevant tables.

Usage:
    python scripts/update_llm_judge_results.py
    python scripts/update_llm_judge_results.py --dry-run   # print diff, no write
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BASE = Path("runs/baseline")
PLAN_PATH = Path("docs/EXPERIMENT_PLAN.md")

# Map dir_name → (model_label, method_label)
MODEL_MAP: dict[str, tuple[str, str]] = {
    "internvl3_5-2b_img256_tok128":             ("InternVL3.5-2B", "Resize"),
    "native_internvl35-2b":                     ("InternVL3.5-2B", "Native"),
    "panoadapt_internvl35-2b":                  ("InternVL3.5-2B", "+DenseCL"),
    "panoadapt_vicreg_pairwise_internvl35-2b":  ("InternVL3.5-2B", "+VICReg-pw 50%"),
    "qwen25-vl-3b_img256_tok128":               ("Qwen2.5-VL-3B",  "Resize"),
    "native_qwen25-vl-3b":                      ("Qwen2.5-VL-3B",  "Native"),
    "panoadapt_qwen25-vl-3b":                   ("Qwen2.5-VL-3B",  "+DenseCL"),
    "panoadapt_vicreg_pairwise_qwen25-vl-3b":   ("Qwen2.5-VL-3B",  "+VICReg-pw 50%"),
    "gemma3-4b_img256_tok128":                  ("Gemma3-4B",       "Resize"),
    "native_gemma3-4b":                         ("Gemma3-4B",       "Native"),
    "panoadapt_gemma3-4b":                      ("Gemma3-4B",       "+DenseCL"),
}


def load_score(dir_name: str) -> tuple[float | None, str]:
    """Return (mean_weighted_score, status_str).

    Returns:
        (score, judge_model)   if valid stats exist
        (None, '🔄')           if stats file missing (still running)
        (None, '❌')           if stats exist but std=0 (API failure)
    """
    stats_path = BASE / dir_name / "llm_judge" / "predictions_judge_scores.stats.json"
    if not stats_path.exists():
        return None, "🔄"
    try:
        d = json.loads(stats_path.read_text())
        std = d.get("std_weighted_score", 0.0)
        if std < 0.01:   # all-3.0 default → API failure
            return None, "❌"
        score = d.get("mean_weighted_score")
        judge = d.get("model", "unknown")
        return float(score), judge
    except Exception:
        return None, "❌"


def fmt_score(score: float | None, status: str) -> str:
    if score is None:
        return status
    return f"{score:.4f}"


def update_llm_judge_section(text: str, scores: dict[str, tuple[float | None, str]]) -> str:
    """Update the 'LLM-Judge 현황' table in EXPERIMENT_PLAN.md."""
    # Build new table rows
    rows: list[str] = []
    for dir_name, (model_label, method_label) in MODEL_MAP.items():
        score, status = scores[dir_name]
        score_str = fmt_score(score, status)
        judge_model = f"gpt-5.2" if status not in ("🔄", "❌") else status
        if score is not None:
            # show model used
            _, raw_status = scores[dir_name]
            judge_model = raw_status   # reuse "status" as judge model name from load_score
        rows.append(f"| {model_label} | {method_label} | {score_str} | {judge_model} |")

    # Match the existing table header + separator
    section_re = re.compile(
        r"(### LLM-Judge 현황.*?)\n"
        r"(\|.*?\|.*?\n"    # header
        r"\|[-: |]+\|\n)"   # separator
        r"((?:\|.*?\|\n)*)",  # data rows
        re.DOTALL,
    )
    header_block = (
        "| 모델 | 방법 | LLM-Judge ↑ | 상태 |\n"
        "|------|------|:---:|:---:|\n"
    )
    rows_block = "\n".join(rows) + "\n"

    def replacer(m: re.Match) -> str:  # type: ignore[type-arg]
        return m.group(1) + "\n" + header_block + rows_block

    new_text, n = section_re.subn(replacer, text)
    if n == 0:
        print("[WARN] Could not find LLM-Judge 현황 section to update")
    return new_text


def update_native_table(text: str, scores: dict[str, tuple[float | None, str]]) -> str:
    """Update LLM-Judge column in Section 3 Native Baseline table."""
    mapping = {
        "InternVL3.5-2B": scores.get("native_internvl35-2b", (None, "🔄"))[0],
        "Qwen2.5-VL-3B":  scores.get("native_qwen25-vl-3b", (None, "🔄"))[0],
        "Gemma3-4B":       scores.get("native_gemma3-4b", (None, "🔄"))[0],
    }

    def replace_row(m: re.Match) -> str:  # type: ignore[type-arg]
        line = m.group(0)
        for name, score in mapping.items():
            if name in line and "LLM-Judge" not in line:
                # line ends with | 상태 | — add LLM-Judge before 상태
                pass
            if name in line:
                # Replace trailing 🔄 or existing score in LLM-Judge column
                score_str = fmt_score(score, "🔄")
                line = re.sub(r"\| (🔄|[\d.]+) \| (✅|⏳)", f"| {score_str} | \\2", line)
        return line

    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        for name, score in mapping.items():
            if f"| {name}" in line or f"| **{name}**" in line:
                score_str = fmt_score(score, "🔄")
                line = re.sub(r"\| (🔄|[\d.]+) \| (✅|⏳|🔄)",
                               f"| {score_str} | \\2", line)
                break
        out.append(line)
    return "\n".join(out)


def update_densecl_table(text: str, scores: dict[str, tuple[float | None, str]]) -> str:
    """Update LLM-Judge column in Section 4.2 DenseCL table."""
    mapping = {
        "InternVL3.5-2B": scores.get("panoadapt_internvl35-2b", (None, "🔄"))[0],
        "Qwen2.5-VL-3B":  scores.get("panoadapt_qwen25-vl-3b", (None, "🔄"))[0],
        "Gemma3-4B":       scores.get("panoadapt_gemma3-4b", (None, "🔄"))[0],
    }
    lines = text.split("\n")
    out: list[str] = []
    # Track which table we're in by looking for the DenseCL header
    in_densecl_table = False
    for line in lines:
        if "| Model | Native CIDEr | PanoAdapt CIDEr" in line and "LLM-Judge" in line:
            in_densecl_table = True
        elif in_densecl_table and line.startswith("---"):
            in_densecl_table = False
        elif in_densecl_table:
            for name, score in mapping.items():
                if f"| {name}" in line:
                    score_str = fmt_score(score, "🔄")
                    line = re.sub(r"\| 🔄 \|$", f"| {score_str} |", line)
                    line = re.sub(r"\| ([\d.]+) \|$", f"| {score_str} |", line)
                    break
        out.append(line)
    return "\n".join(out)


def update_vicreg_table(text: str, scores: dict[str, tuple[float | None, str]]) -> str:
    """Update LLM-Judge column in Section 4.3 VICReg-pairwise table."""
    mapping = {
        "50%_InternVL": scores.get("panoadapt_vicreg_pairwise_internvl35-2b", (None, "🔄"))[0],
        "50%_Qwen":     scores.get("panoadapt_vicreg_pairwise_qwen25-vl-3b", (None, "🔄"))[0],
    }
    lines = text.split("\n")
    out: list[str] = []
    in_vicreg_table = False
    for line in lines:
        if "| Model | Overlap | Native CIDEr" in line and "LLM-Judge" in line:
            in_vicreg_table = True
        elif in_vicreg_table and line.startswith("---"):
            in_vicreg_table = False
        elif in_vicreg_table:
            if "InternVL3.5-2B" in line and "50%" in line:
                score = mapping["50%_InternVL"]
                score_str = fmt_score(score, "🔄")
                line = re.sub(r"\| (🔄|[\d.]+) \|$", f"| {score_str} |", line)
            elif "Qwen2.5-VL-3B" in line:
                score = mapping["50%_Qwen"]
                score_str = fmt_score(score, "🔄")
                line = re.sub(r"\| (🔄|[\d.]+) \|$", f"| {score_str} |", line)
        out.append(line)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update LLM-Judge scores in EXPERIMENT_PLAN.md")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing")
    args = parser.parse_args()

    # Load all scores
    scores: dict[str, tuple[float | None, str]] = {}
    print("\n── LLM-Judge Score Summary ──────────────────────────")
    for dir_name, (model_label, method_label) in MODEL_MAP.items():
        score, status = load_score(dir_name)
        scores[dir_name] = (score, status)
        score_str = f"{score:.4f}" if score is not None else status
        print(f"  {model_label:<20s} / {method_label:<14s}  →  {score_str}")
    print("─────────────────────────────────────────────────────\n")

    if args.dry_run:
        print("[dry-run] No file changes made.")
        return

    text = PLAN_PATH.read_text(encoding="utf-8")
    text = update_llm_judge_section(text, scores)
    text = update_native_table(text, scores)
    text = update_densecl_table(text, scores)
    text = update_vicreg_table(text, scores)
    PLAN_PATH.write_text(text, encoding="utf-8")
    print(f"Updated: {PLAN_PATH}")


if __name__ == "__main__":
    main()
