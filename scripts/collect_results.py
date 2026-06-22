#!/usr/bin/env python3
"""Collect and display all CORA + baseline evaluation results in a single table.

Usage:
  python scripts/collect_results.py              # Print markdown table
  python scripts/collect_results.py --latex       # Print LaTeX table
  python scripts/collect_results.py --csv out.csv # Save as CSV
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ── Configuration ──

BASELINE_DIR = Path("runs/baseline")
CORA_OUTPUT_DIR = Path("outputs")

# Baseline models (dir_name → (display_name, params, track))
# Track A: Native processing (8 models)
TRACK_A_NATIVE = {
    "native_qwen25-vl-3b":    ("Qwen2.5-VL-3B",    "3B", "Native"),
    "native_qwen2-vl-2b":     ("Qwen2-VL-2B",      "2B", "Native"),
    "native_internvl35-2b":   ("InternVL3.5-2B",   "2B", "Native"),
    "native_internvl35-1b":   ("InternVL3.5-1B",   "1B", "Native"),
    "native_gemma3-4b":       ("Gemma3-4B",        "4B", "Native"),
    "native_blip2-2.7b":      ("BLIP2-OPT-2.7B",   "2.7B", "Native"),
    "native_internvl25-4b":   ("InternVL2.5-4B",   "4B", "Native"),
    "native_internvl25-2b":   ("InternVL2.5-2B",   "2B", "Native"),
}

# Track B: PanoAdapt experiments (3 models)
TRACK_B_PANOADAPT = {
    "panoadapt_qwen25-vl-3b":   ("Qwen2.5-VL-3B + PanoAdapt(DenseCL)",   "3B", "PanoAdapt"),
    "panoadapt_internvl35-2b":  ("InternVL3.5-2B + PanoAdapt(DenseCL)",  "2B", "PanoAdapt"),
    "panoadapt_gemma3-4b":      ("Gemma3-4B + PanoAdapt(DenseCL)",       "4B", "PanoAdapt"),
}

# Track B': PanoAdapt VICReg-pairwise experiments
TRACK_B_VICREG_PW = {
    "panoadapt_vicreg_pairwise_internvl35-2b":             ("InternVL3.5-2B + VICReg-pw(50%)",   "2B", "VICReg-pw"),
    "panoadapt_vicreg_pairwise_internvl35-2b_25overlap":   ("InternVL3.5-2B + VICReg-pw(25%)",   "2B", "VICReg-pw"),
    "panoadapt_vicreg_pairwise_qwen25-vl-3b":              ("Qwen2.5-VL-3B + VICReg-pw(50%)",    "3B", "VICReg-pw"),
    "panoadapt_vicreg_pairwise_gemma3-4b":                 ("Gemma3-4B + VICReg-pw(50%)",        "4B", "VICReg-pw"),
}

# Track M: Multiview strategy comparison (Qwen2.5-VL-3B only)
# These have nested output dirs: runs/baseline/{dir_name}/qwen25-vl-3b/eval/metrics.json
TRACK_M_MULTIVIEW = {
    # (dir_name, sub_dir, display_name, params, views)
    "dynamic_qwen25-vl-3b":     ("qwen25-vl-3b", "Qwen2.5-VL-3B (native/dynamic)",    "3B", "1 view"),
    "cubemap_qwen25-vl-3b":     ("qwen25-vl-3b", "Qwen2.5-VL-3B (cubemap 90°)",       "3B", "5 views"),
    "anyres_e2p_qwen25-vl-3b":  ("qwen25-vl-3b", "Qwen2.5-VL-3B (anyres_e2p 45°)",    "3B", "9 views"),
}
# Note: resize (256²) = qwen25-vl-3b_img256_tok128 in BASELINES dict

# Old baselines (fixed 256² resolution, kept for reference)
BASELINES = {
    "gemma3-4b_img256_tok128":       ("Gemma3-4B",       "4B", "Native"),
    "internvl3_5-2b_img256_tok128":  ("InternVL3.5-2B",  "2B", "Native"),
    "qwen25-vl-3b_img256_tok128":    ("Qwen2.5-VL-3B",   "3B", "Native"),
    "qwen2-vl-2b_img256_tok128":     ("Qwen2-VL-2B",     "2B", "Native"),
    "blip2-opt-2.7b_img256_tok128":  ("BLIP2-OPT-2.7B",  "2.7B", "Native"),
    "internvl3_5-1b_img256_tok128":  ("InternVL3.5-1B",  "1B", "Native"),
}

# CORA experiments (dir_name → display_name, params)
CORA_EXPERIMENTS = {
    "cora_contrastive":              ("CORA-InfoNCE",              "615M"),
    "cora_vicreg_batchwise":         ("CORA-VICReg-Batch",         "615M"),
    "cora_vicreg":                   ("CORA-VICReg-Pair",          "615M"),
    "cora_densecl":                  ("CORA-DenseCL",              "615M"),
    "cora_vicreg_overlap05":         ("CORA-VICReg-Pair-0.5",      "615M"),
    "cora_densecl_overlap05":        ("CORA-DenseCL-0.5",          "615M"),
}

# CORA ablation experiments (different crop strategies)
CORA_ABLATION = {
    "cora_ablation_resize":     ("CORA-DenseCL (resize)",   "615M"),
    "cora_ablation_cubemap":    ("CORA-DenseCL (cubemap)",  "615M"),
    "cora_ablation_e2p":        ("CORA-DenseCL (e2p)",      "615M"),
}

# Metrics to display (key in JSON → display_name)
METRICS = [
    ("bleu_4",  "BLEU-4"),
    ("meteor",  "METEOR"),
    ("rouge_l", "ROUGE-L"),
    ("cider",   "CIDEr"),
    ("spice",   "SPICE"),
    ("llm_judge", "LLM-Judge"),
]


# ── Data Loading ──

def load_baseline_metrics(name: str) -> dict[str, object] | None:
    """Load metrics from baseline eval directory.

    Prefers eval_v2 (re-evaluation with fixes) over eval/ when available.
    """
    # Prefer re-evaluation results (eval_v2) over original
    metrics_path = BASELINE_DIR / name / "eval_v2" / "metrics.json"
    if not metrics_path.exists():
        metrics_path = BASELINE_DIR / name / "eval" / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        data = json.load(f)
    # Convert NaN strings/floats to None for safe display
    return {k: (None if v != v else v) if isinstance(v, float) else v for k, v in data.items()}


def load_cora_metrics(name: str) -> dict[str, object] | None:
    """Load metrics from CORA output directory.

    Preference order: _greedy > _v2 > base  (greedy decoding is the fair comparison).
    """
    for suffix in ("_greedy", "_v2", ""):
        base = CORA_OUTPUT_DIR / f"{name}{suffix}"
        metrics_json = base / "metrics.json"
        if metrics_json.exists():
            with open(metrics_json) as f:
                data = json.load(f)
            return {k: (None if isinstance(v, float) and v != v else v) for k, v in data.items()}
        metrics_csv = base / "metrics.csv"
        if metrics_csv.exists():
            import csv
            with open(metrics_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    return {k: float(v) for k, v in row.items() if k != "experiment"}
    return None

def load_multiview_metrics(dir_name: str, sub_dir: str) -> dict[str, object] | None:
    """Load metrics from multiview eval directory with nested sub_dir."""
    metrics_path = BASELINE_DIR / dir_name / sub_dir / "eval" / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        data = json.load(f)
    return {k: (None if v != v else v) if isinstance(v, float) else v for k, v in data.items()}


def load_llm_judge_score(name: str) -> float | None:
    """Load LLM-Judge weighted score from baseline eval directory.

    Searches for stats JSON in {BASELINE_DIR}/{name}/llm_judge/ directory.
    Returns the 'mean_weighted_score' (primary) or 'mean_overall' (fallback).
    """
    judge_dir = BASELINE_DIR / name / "llm_judge"
    if not judge_dir.exists():
        return None
    # Try multiple filename patterns
    for fname in ("predictions_judge_scores.stats.json", "judge_scores.stats.json"):
        judge_path = judge_dir / fname
        if judge_path.exists():
            try:
                with open(judge_path) as f:
                    data = json.load(f)
                # Prefer weighted_score, fallback to overall
                return data.get("mean_weighted_score") or data.get("mean_overall")
            except (json.JSONDecodeError, KeyError):
                continue
    return None

# ── Table Formatting ──

def format_value(val: float | None, is_best: bool = False) -> str:
    """Format metric value with optional bold for best."""
    if val is None:
        return "—"
    s = f"{val:.4f}"
    if is_best:
        return f"**{s}**"
    return s


def find_best_per_metric(rows: list[dict[str, object]]) -> dict[str, float]:
    """Find the best value for each metric across all rows."""
    best: dict[str, float] = {}
    for metric_key, _ in METRICS:
        values = []
        for r in rows:
            if r["metrics"] is not None:
                m = r["metrics"]
                if isinstance(m, dict):
                    v = m.get(metric_key)
                    if v is not None:
                        values.append(v)
        values = [v for v in values if isinstance(v, (int, float)) and v > 0]
        if values:
            best[metric_key] = max(values)
    return best


def print_markdown_table(rows: list[dict[str, object]], best: dict[str, float]) -> None:
    """Print results as markdown table."""
    header = "| Model | Track | Params |"
    separator = "|-------|:---:|:---:|"
    for _, display in METRICS:
        header += f" {display} ↑ |"
        separator += ":---:|"

    print()
    print(header)
    print(separator)

    for row in rows:
        line = f"| {row['name']} | {row['track']} | {row['params']} |"
        for metric_key, _ in METRICS:
            if row["metrics"] is None:
                val_str = "—"
            else:
                m = row["metrics"]
                if isinstance(m, dict):
                    val = m.get(metric_key)
                    is_best = val is not None and isinstance(val, (int, float)) and val > 0 and best.get(metric_key) == val
                    val_str = format_value(val, is_best)
                else:
                    val_str = "—"
            line += f" {val_str} |"
        print(line)

    print()


def print_latex_table(rows: list[dict[str, object]], best: dict[str, float]) -> None:
    """Print results as LaTeX table."""
    n_metrics = len(METRICS)
    col_spec = "l c c" + " c" * n_metrics

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Comparison of native VLM processing vs PanoAdapt on QuIC-360 test set (5349 samples).}")
    print(r"\label{tab:results}")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print(r"\toprule")

    header = "Model & Track & Params"
    for _, display in METRICS:
        header += f" & {display} $\\uparrow$"
    header += r" \\"
    print(header)
    print(r"\midrule")

    def _get_dir_name(r: dict[str, object]) -> str:
        dn = r.get("dir_name")
        return str(dn) if dn is not None else ""

    def _get_track(r: dict[str, object]) -> str:
        t = r.get("track")
        return str(t) if t is not None else ""

    def _get_metrics_dict(r: dict[str, object]) -> dict[str, object] | None:
        m = r.get("metrics")
        return m if isinstance(m, dict) else None

    track_a_rows = [r for r in rows if _get_track(r) == "Native" and "img256" not in _get_dir_name(r)]
    track_b_rows = [r for r in rows if _get_track(r) == "PanoAdapt"]
    track_bv_rows = [r for r in rows if _get_track(r) == "VICReg-pw"]
    track_m_rows = [r for r in rows if _get_track(r).startswith("Multiview")]
    old_baseline_rows = [r for r in rows if _get_track(r) == "Native" and "img256" in _get_dir_name(r)]
    cora_rows = [r for r in rows if r.get("is_cora") and _get_track(r) == "CORA"]
    cora_abl_rows = [r for r in rows if r.get("is_cora") and _get_track(r) == "CORA-ablation"]

    for row in track_a_rows:
        line = f"{row['name']} & {row['track']} & {row['params']}"
        metrics = _get_metrics_dict(row)
        for metric_key, _ in METRICS:
            if metrics is None:
                line += " & —"
            else:
                val = metrics.get(metric_key)
                if val is None or val == 0:
                    line += " & —"
                else:
                    is_best = best.get(metric_key) == val
                    s = f"{val:.4f}"
                    line += f" & \\textbf{{{s}}}" if is_best else f" & {s}"
        line += r" \\"
        print(line)

    if track_b_rows:
        print(r"\midrule")
        for row in track_b_rows:
            line = f"{row['name']} & {row['track']} & {row['params']}"
            metrics = _get_metrics_dict(row)
            for metric_key, _ in METRICS:
                if metrics is None:
                    line += " & —"
                else:
                    val = metrics.get(metric_key)
                    if val is None or val == 0:
                        line += " & —"
                    else:
                        is_best = best.get(metric_key) == val
                        s = f"{val:.4f}"
                        line += f" & \\textbf{{{s}}}" if is_best else f" & {s}"
            line += r" \\"
            print(line)

    if old_baseline_rows:
        print(r"\midrule")
        for row in old_baseline_rows:
            line = f"{row['name']} & {row['track']} & {row['params']}"
            metrics = _get_metrics_dict(row)
            for metric_key, _ in METRICS:
                if metrics is None:
                    line += " & —"
                else:
                    val = metrics.get(metric_key)
                    if val is None or val == 0:
                        line += " & —"
                    else:
                        is_best = best.get(metric_key) == val
                        s = f"{val:.4f}"
                        line += f" & \\textbf{{{s}}}" if is_best else f" & {s}"
            line += r" \\"
            print(line)

    for section_rows in (track_bv_rows, track_m_rows, old_baseline_rows, cora_rows, cora_abl_rows):
        if section_rows:
            print(r"\midrule")
            for row in section_rows:
                line = f"{row['name']} & {row['track']} & {row['params']}"
                metrics = _get_metrics_dict(row)
                for metric_key, _ in METRICS:
                    if metrics is None:
                        line += " & —"
                    else:
                        val = metrics.get(metric_key)
                        if val is None or val == 0:
                            line += " & —"
                        else:
                            is_best = best.get(metric_key) == val
                            s = f"{val:.4f}"
                            line += f" & \\textbf{{{s}}}" if is_best else f" & {s}"
                line += r" \\"
                print(line)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CORA experiment results")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table")
    parser.add_argument("--csv", type=str, default=None, help="Save to CSV file")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []

    print("Scanning Track A (Native) results...")
    for dir_name, (display_name, params, track) in TRACK_A_NATIVE.items():
        metrics = load_baseline_metrics(dir_name)
        llm_judge = load_llm_judge_score(dir_name)
        if metrics is not None and llm_judge is not None:
            metrics["llm_judge"] = llm_judge
        status = "✅" if metrics else "⏳"
        print(f"  {status} {display_name}: {BASELINE_DIR / dir_name / 'eval'}")
        rows.append({
            "name": display_name,
            "params": params,
            "track": track,
            "metrics": metrics,
            "is_cora": False,
            "dir_name": dir_name,
        })

    print("\nScanning Track B (PanoAdapt) results...")
    for dir_name, (display_name, params, track) in TRACK_B_PANOADAPT.items():
        metrics = load_baseline_metrics(dir_name)
        llm_judge = load_llm_judge_score(dir_name)
        if metrics is not None and llm_judge is not None:
            metrics["llm_judge"] = llm_judge
        status = "✅" if metrics else "⏳"
        print(f"  {status} {display_name}: {BASELINE_DIR / dir_name / 'eval'}")
        rows.append({
            "name": display_name,
            "params": params,
            "track": track,
            "metrics": metrics,
            "is_cora": False,
            "dir_name": dir_name,
        })

    print("\nScanning Track B' (VICReg-pairwise) results...")
    for dir_name, (display_name, params, track) in TRACK_B_VICREG_PW.items():
        metrics = load_baseline_metrics(dir_name)
        llm_judge = load_llm_judge_score(dir_name)
        if metrics is not None and llm_judge is not None:
            metrics["llm_judge"] = llm_judge
        status = "✅" if metrics else "⏳"
        print(f"  {status} {display_name}: {BASELINE_DIR / dir_name / 'eval'}")
        rows.append({
            "name": display_name,
            "params": params,
            "track": track,
            "metrics": metrics,
            "is_cora": False,
            "dir_name": dir_name,
        })

    print("\nScanning Track M (Multiview) results...")
    for dir_name, (sub_dir, display_name, params, views) in TRACK_M_MULTIVIEW.items():
        metrics = load_multiview_metrics(dir_name, sub_dir)
        llm_judge = load_llm_judge_score(dir_name)
        if metrics is not None and llm_judge is not None:
            metrics["llm_judge"] = llm_judge
        status = "✅" if metrics else "⏳"
        print(f"  {status} {display_name} [{views}]: {BASELINE_DIR / dir_name / sub_dir / 'eval'}")
        rows.append({
            "name": display_name,
            "params": params,
            "track": f"Multiview ({views})",
            "metrics": metrics,
            "is_cora": False,
            "dir_name": dir_name,
        })

    print("\nScanning old baseline results...")
    for dir_name, (display_name, params, track) in BASELINES.items():
        metrics = load_baseline_metrics(dir_name)
        llm_judge = load_llm_judge_score(dir_name)
        if metrics is not None and llm_judge is not None:
            metrics["llm_judge"] = llm_judge
        status = "✅" if metrics else "❌"
        print(f"  {status} {display_name}: {BASELINE_DIR / dir_name / 'eval'}")
        rows.append({
            "name": display_name,
            "params": params,
            "track": track,
            "metrics": metrics,
            "is_cora": False,
            "dir_name": dir_name,
        })

    print("\nScanning CORA results...")
    for dir_name, (display_name, params) in CORA_EXPERIMENTS.items():
        metrics = load_cora_metrics(dir_name)
        status = "✅" if metrics else "⏳"
        print(f"  {status} {display_name}: {CORA_OUTPUT_DIR / dir_name}")
        rows.append({
            "name": display_name,
            "params": params,
            "track": "CORA",
            "metrics": metrics,
            "is_cora": True,
            "dir_name": dir_name,
        })

    print("\nScanning CORA ablation results...")
    for dir_name, (display_name, params) in CORA_ABLATION.items():
        metrics = load_cora_metrics(dir_name)
        status = "✅" if metrics else "⏳"
        print(f"  {status} {display_name}: {CORA_OUTPUT_DIR / dir_name}")
        rows.append({
            "name": display_name,
            "params": params,
            "track": "CORA-ablation",
            "metrics": metrics,
            "is_cora": True,
            "dir_name": dir_name,
        })

    best = find_best_per_metric(rows)

    if args.latex:
        print_latex_table(rows, best)
    elif args.csv:
        import csv as csv_mod
        out_path = Path(args.csv)
        with open(out_path, "w", newline="") as f:
            writer = csv_mod.writer(f)
            header = ["model", "track", "params"] + [mk for mk, _ in METRICS]
            writer.writerow(header)
            for row in rows:
                vals = []
                for mk, _ in METRICS:
                    m = row.get("metrics")
                    v = None
                    if isinstance(m, dict):
                        v = m.get(mk)
                    vals.append(f"{v:.6f}" if isinstance(v, (int, float)) else "")
                writer.writerow([row["name"], row["track"], row["params"]] + vals)
        print(f"\nSaved to {out_path}")
    else:
        print_markdown_table(rows, best)


if __name__ == "__main__":
    main()
