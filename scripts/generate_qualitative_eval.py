#!/usr/bin/env python3
"""Generate qualitative evaluation markdown from LLM judge scores.

Reads 4 model variants' judge CSVs, selects 10 diverse samples by difficulty,
copies panoramic images, and writes a formatted markdown document.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────

BASE = Path("runs/baseline")
OUT_DIR = Path("docs/qualitative_eval")

MODELS = {
    "Resize": "baseline_internvl35-2b_img256",
    "Native": "trackA_internvl35-2b_native",
    "+DenseCL": "trackB_internvl35-2b_densecl50",
    "+VICReg-pw": "trackB_internvl35-2b_vicreg50",
}

# Difficulty thresholds (on 100-point weighted_score average)
HARD_UPPER = 58
EASY_LOWER = 78

# How many to pick per band
N_HARD, N_MED, N_EASY = 3, 4, 3

SCORE_COLS = [
    "spatial_coherence",
    "query_relevance",
    "factual_accuracy",
    "completeness",
    "fluency",
]

JUDGE_MODEL_NAME = "gpt-5.2"


# ── Data loading ───────────────────────────────────────────────────────────


def load_judge_csvs() -> dict[str, pd.DataFrame]:
    """Load judge score CSVs for all 4 model variants."""
    frames: dict[str, pd.DataFrame] = {}
    for label, dirname in MODELS.items():
        path = BASE / dirname / "llm_judge" / "predictions_judge_scores.csv"
        df = pd.read_csv(path)
        frames[label] = df
    return frames


def merge_scores(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge 4 model judge CSVs into a single wide DataFrame.

    Columns: sample_id, image_path, query, reference,
             {model}_prediction, {model}_weighted_score, {model}_{dim}, ...
             avg_score
    """
    base_key = list(frames.keys())[0]
    merged = frames[base_key][["sample_id", "image_path", "query", "reference"]].copy()

    for label, df in frames.items():
        safe = label.replace("+", "plus_").replace("-", "_").replace(" ", "_")
        merged[f"{safe}_prediction"] = df["prediction"]
        merged[f"{safe}_weighted_score"] = df["weighted_score"]
        merged[f"{safe}_overall_rationale"] = df["overall_rationale"]
        for col in SCORE_COLS:
            merged[f"{safe}_{col}"] = df[col]

    # Average weighted score across 4 models
    ws_cols = [c for c in merged.columns if c.endswith("_weighted_score")]
    merged["avg_score"] = merged[ws_cols].mean(axis=1)
    return merged


# ── Sample selection ───────────────────────────────────────────────────────


def select_diverse(
    pool: pd.DataFrame, n: int, seed: int = 42
) -> pd.DataFrame:
    """Select n samples from pool ensuring query and image diversity.

    Strategy: sort by avg_score, split into n equal bins, pick 1 per bin
    ensuring unique queries and unique images.
    """
    pool = pool.copy().sort_values("avg_score").reset_index(drop=True)

    if len(pool) <= n:
        return pool

    # Bin-based stratified selection
    bin_size = len(pool) // n
    selected_queries: set[str] = set()
    selected_images: set[str] = set()
    selected: list[int] = []

    for i in range(n):
        start = i * bin_size
        end = start + bin_size if i < n - 1 else len(pool)
        candidates = pool.iloc[start:end]

        # Filter for diversity
        for _, row in candidates.iterrows():
            q = row["query"].lower().strip()
            img = row["image_path"]
            if q not in selected_queries and img not in selected_images:
                selected.append(row.name)
                selected_queries.add(q)
                selected_images.add(img)
                break
        else:
            # Fallback: just pick first unused image
            for _, row in candidates.iterrows():
                if row["image_path"] not in selected_images:
                    selected.append(row.name)
                    selected_images.add(row["image_path"])
                    selected_queries.add(row["query"].lower().strip())
                    break

    return pool.loc[selected].reset_index(drop=True)


def select_samples(merged: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split into hard/medium/easy, select diverse samples from each."""
    hard = merged[merged["avg_score"] < HARD_UPPER].copy()
    medium = merged[
        (merged["avg_score"] >= HARD_UPPER) & (merged["avg_score"] < EASY_LOWER)
    ].copy()
    easy = merged[merged["avg_score"] >= EASY_LOWER].copy()

    print(f"Pool sizes — Hard: {len(hard)}, Medium: {len(medium)}, Easy: {len(easy)}")

    bands = [
        ("hard", select_diverse(hard, N_HARD)),
        ("medium", select_diverse(medium, N_MED)),
        ("easy", select_diverse(easy, N_EASY)),
    ]
    return bands


# ── Image handling ─────────────────────────────────────────────────────────


def copy_images(bands: list[tuple[str, pd.DataFrame]]) -> dict[int, str]:
    """Copy selected images to docs/qualitative_eval/ as img_NN.jpg."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old images
    for old in OUT_DIR.glob("img_*.jpg"):
        old.unlink()

    idx = 1
    sample_to_img: dict[int, str] = {}

    for _band_name, df in bands:
        for _, row in df.iterrows():
            src = Path(row["image_path"])
            dst_name = f"img_{idx:02d}.jpg"
            dst = OUT_DIR / dst_name

            if src.exists():
                shutil.copy2(src, dst)
            else:
                print(f"WARNING: Image not found: {src}")

            sample_to_img[row["sample_id"]] = dst_name
            idx += 1

    return sample_to_img


# ── Markdown generation ────────────────────────────────────────────────────


def _truncate(text: str, max_len: int = 300) -> str:
    """Truncate text, preserving sentence boundaries where possible."""
    text = str(text).strip()
    if len(text) <= max_len:
        return text
    # Try to cut at last sentence boundary
    cut = text[:max_len]
    last_period = cut.rfind(".")
    if last_period > max_len * 0.5:
        return cut[: last_period + 1]
    return cut.rstrip() + " …"


def _get_safe(label: str) -> str:
    return label.replace("+", "plus_").replace("-", "_").replace(" ", "_")


def generate_markdown(
    bands: list[tuple[str, pd.DataFrame]],
    sample_to_img: dict[int, str],
) -> str:
    """Generate the full markdown document."""
    lines: list[str] = []

    # ── Header
    total = sum(len(df) for _, df in bands)
    lines.append("# 정성 평가 샘플 — QuIC-360 Panoramic Captioning\n")
    lines.append(f"> **모델**: InternVL3.5-2B × 4 방법 (Resize / Native / +DenseCL / +VICReg-pw)  ")
    lines.append(f"> **Judge**: {JUDGE_MODEL_NAME}, 이미지 포함, 100점 만점  ")
    lines.append(f"> **샘플**: 300개 테스트 풀에서 난이도별 {total}개 선별 (상 {N_HARD} / 중 {N_MED} / 하 {N_EASY})  ")
    lines.append("> **선별 기준**: 4모델 평균 점수 기준 계층적 샘플링 + 쿼리·이미지 다양성 확보")
    lines.append("")
    lines.append("> **점수 산출**: Spatial(30%) + Query Relevance(25%) + Factual(20%) + Completeness(15%) + Fluency(10%), 각 1–5점 → 가중합 × 20 = 100점 만점")
    lines.append("")

    band_labels = {
        "hard": ("🔴 Hard", "모델이 어려워하는 케이스", f"avg < {HARD_UPPER}"),
        "medium": ("🟡 Medium", "중간 난이도 케이스", f"avg {HARD_UPPER}–{EASY_LOWER}"),
        "easy": ("🟢 Easy", "모델이 잘 처리하는 케이스", f"avg ≥ {EASY_LOWER}"),
    }

    sample_num = 1
    model_labels = list(MODELS.keys())

    # Track all samples for summary table
    all_samples: list[dict] = []

    for band_name, df in bands:
        emoji, desc, cond = band_labels[band_name]
        lines.append("---\n")
        lines.append(f"## {emoji} — {desc} ({cond})\n")

        for _, row in df.iterrows():
            avg = row["avg_score"]
            lines.append(f"### Sample {sample_num} &nbsp; avg score: **{avg:.0f} / 100**\n")
            lines.append(f"**Query**: `{row['query']}`\n")

            img_name = sample_to_img.get(row["sample_id"], "missing.jpg")
            lines.append(f"![{img_name}]({img_name})\n")

            ref_text = str(row["reference"]).strip()
            lines.append(f"**Reference**: {ref_text}\n")

            # Table header
            lines.append("| 방법 | 응답 | Judge | Spatial | Query | Factual | Complete | Fluency |")
            lines.append("|:-----|:-----|:-----:|:-------:|:-----:|:-------:|:--------:|:-------:|")

            best_label = ""
            best_score = -1
            best_rationale = ""

            for label in model_labels:
                safe = _get_safe(label)
                pred = str(row[f"{safe}_prediction"]).strip()
                ws = row[f"{safe}_weighted_score"]
                sp = int(row[f"{safe}_spatial_coherence"])
                qr = int(row[f"{safe}_query_relevance"])
                fa = int(row[f"{safe}_factual_accuracy"])
                co = int(row[f"{safe}_completeness"])
                fl = int(row[f"{safe}_fluency"])
                rationale = str(row[f"{safe}_overall_rationale"]).strip()

                lines.append(
                    f"| **{label}** | {pred} | **{ws:.0f}** | {sp} | {qr} | {fa} | {co} | {fl} |"
                )

                if ws > best_score:
                    best_score = ws
                    best_label = label
                    best_rationale = rationale

            lines.append("")

            # Judge comment from highest-scoring method
            lines.append(
                f"> 💬 **Judge 코멘트** ({best_label}): {best_rationale}"
            )
            lines.append("")

            all_samples.append({
                "num": sample_num,
                "band": band_name,
                "query": row["query"],
                "avg": avg,
            })

            sample_num += 1

    # ── Summary section
    lines.append("---\n")
    lines.append("## 📊 종합 분석\n")

    # Summary table
    lines.append("### 샘플별 요약\n")
    lines.append("| # | 난이도 | Query | Avg Score | Resize | Native | +DenseCL | +VICReg-pw |")
    lines.append("|:-:|:------:|:------|:---------:|:------:|:------:|:--------:|:----------:|")

    sample_num = 1
    for band_name, df in bands:
        for _, row in df.iterrows():
            band_emoji = {"hard": "🔴", "medium": "🟡", "easy": "🟢"}[band_name]
            scores = []
            for label in model_labels:
                safe = _get_safe(label)
                scores.append(f"{row[f'{safe}_weighted_score']:.0f}")

            lines.append(
                f"| {sample_num} | {band_emoji} | `{row['query']}` | **{row['avg_score']:.0f}** | "
                + " | ".join(scores)
                + " |"
            )
            sample_num += 1

    lines.append("")

    # Method averages across selected samples
    lines.append("### 방법별 평균 (선별 샘플 기준)\n")
    lines.append("| 방법 | Avg Score | Spatial | Query | Factual | Complete | Fluency |")
    lines.append("|:-----|:---------:|:-------:|:-----:|:-------:|:--------:|:-------:|")

    for label in model_labels:
        safe = _get_safe(label)
        ws_vals, sp_vals, qr_vals, fa_vals, co_vals, fl_vals = [], [], [], [], [], []

        for _band_name, df in bands:
            for _, row in df.iterrows():
                ws_vals.append(row[f"{safe}_weighted_score"])
                sp_vals.append(row[f"{safe}_spatial_coherence"])
                qr_vals.append(row[f"{safe}_query_relevance"])
                fa_vals.append(row[f"{safe}_factual_accuracy"])
                co_vals.append(row[f"{safe}_completeness"])
                fl_vals.append(row[f"{safe}_fluency"])

        n = len(ws_vals)
        lines.append(
            f"| **{label}** | **{sum(ws_vals)/n:.1f}** | "
            f"{sum(sp_vals)/n:.1f} | {sum(qr_vals)/n:.1f} | "
            f"{sum(fa_vals)/n:.1f} | {sum(co_vals)/n:.1f} | "
            f"{sum(fl_vals)/n:.1f} |"
        )

    lines.append("")

    # Key observations
    lines.append("### 주요 관찰\n")
    lines.append("1. **Hard 케이스 공통 패턴**: 추상적 쿼리(`small objects`, `what they are doing`)에서 "
                 "모델이 reference와 동떨어진 hallucination을 생성하는 경향. "
                 "Spatial/Factual 점수가 특히 낮음.")
    lines.append("2. **Medium 케이스**: 쿼리 관련성(Query Relevance)은 높으나, "
                 "reference의 구체적 디테일을 놓치는 경향. Completeness 점수가 병목.")
    lines.append("3. **Easy 케이스**: 시각적으로 명확한 장면(`weather`, `building`, `location`)에서 "
                 "4개 방법 모두 높은 점수. 방법 간 차이 미미.")
    lines.append("4. **방법 간 비교**: Native/+DenseCL/+VICReg-pw가 Resize 대비 "
                 "Medium 난이도에서 차별화되는 경향. Hard에서는 모든 방법이 고전.")
    lines.append("")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading judge CSVs...")
    frames = load_judge_csvs()

    print("Merging scores...")
    merged = merge_scores(frames)

    print("Selecting samples...")
    bands = select_samples(merged)

    for band_name, df in bands:
        print(f"  {band_name}: {len(df)} samples selected")
        for _, row in df.iterrows():
            print(f"    sid={row['sample_id']:3d}  query={row['query']:<20s}  avg={row['avg_score']:.0f}")

    print("Copying images...")
    sample_to_img = copy_images(bands)

    print("Generating markdown...")
    md = generate_markdown(bands, sample_to_img)

    out_path = OUT_DIR / "qualitative_eval.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Written to {out_path} ({len(md)} chars)")


if __name__ == "__main__":
    main()
