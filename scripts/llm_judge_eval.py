#!/usr/bin/env python3
"""LLM-as-a-Judge evaluation for CORA panoramic VLM predictions.

Evaluates query-based 360° panoramic captioning on five weighted dimensions
(all scored 1–5, JSON output):

  Dimension            Weight   What it measures
  ───────────────────  ──────   ─────────────────────────────────────────────
  spatial_coherence    30 %     Panoramic spatial layout understanding
  query_relevance      25 %     Focus on what the query actually asks
  factual_accuracy     20 %     Object / attribute / relation correctness
  completeness         15 %     Coverage of key reference content
  fluency              10 %     Grammar, structure, naturalness

Two evaluation modes:

  Pointwise   Score each candidate against a reference  (default)
  Pairwise    A/B comparison between two model outputs  (--compare)

Supports:
  Chat Completions API  (gpt-4o, gpt-4.1-mini, gpt-4.1)
  Responses API         (gpt-5-mini, gpt-5.2)

Usage:
  python scripts/llm_judge_eval.py --input predictions.csv
  python scripts/llm_judge_eval.py --input preds.csv --model gpt-4o --batch-by-image
  python scripts/llm_judge_eval.py --input model_a.csv --compare model_b.csv
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Pointwise rubric: 5 dimensions × 1–5, nested JSON output.
# Format fields: {query}, {reference}, {candidate}
JUDGE_PROMPT = """\
You are an expert evaluator for query-based panoramic image captioning.

## Task Description
A 360° panoramic image has been shown to a vision-language model along with a user query.
The model generated a caption in response. Your job is to evaluate the quality of the
generated caption by comparing it to a human-written reference caption.

## Inputs
- **Query**: {query}
- **Reference Caption** (human-written ground truth): {reference}
- **Candidate Caption** (model-generated): {candidate}

## Evaluation Criteria

Evaluate the candidate caption on the following **five dimensions**, each scored from
1 (worst) to 5 (best). After scoring each dimension, provide an **overall score** (1–5).

### 1. Query Relevance (1–5)
Does the caption address the specific query? A 360° image contains many contexts;
the caption should focus on the content specified by the query, not describe the
entire scene indiscriminately.

- **5**: Directly and precisely addresses the query with appropriate focus.
- **4**: Mostly addresses the query with minor off-topic content.
- **3**: Partially addresses the query but includes significant irrelevant content.
- **2**: Loosely related to the query; misses the main intent.
- **1**: Completely ignores the query or describes unrelated content.

### 2. Factual Accuracy (1–5)
Are the objects, attributes, spatial relationships, and scene elements described in the
caption factually correct with respect to the reference? This includes object identity,
color, material, count, and relative positions.

- **5**: All stated facts are correct; no hallucinated content.
- **4**: Minor factual errors that do not affect overall understanding.
- **3**: Some factual errors; mix of correct and incorrect claims.
- **2**: Major factual errors; significant hallucination.
- **1**: Predominantly fabricated or contradictory to the reference.

### 3. Spatial Coherence (1–5)
Does the caption demonstrate a coherent understanding of the spatial layout across the
panoramic scene? This is critical for 360° imagery where content spans the full field
of view and objects may appear in different regions (left, right, behind, etc.).
Pay special attention to:
- Correct use of directional/positional language (e.g., "to the left", "in the distance",
  "behind the viewer", "across the room").
- Consistent spatial relationships (e.g., not placing the same object in contradictory
  locations).
- Awareness that the scene wraps around 360° (no artificial "edges").

- **5**: Spatial descriptions are accurate, consistent, and demonstrate panoramic awareness.
- **4**: Mostly spatially coherent with minor inconsistencies.
- **3**: Some spatial confusion; partially correct layout description.
- **2**: Significant spatial errors; contradictory or implausible layout.
- **1**: No meaningful spatial reasoning; random or flat description.

### 4. Completeness (1–5)
Does the caption cover the key elements mentioned in the reference caption?
A good caption should capture the important objects, actions, and scene characteristics
relevant to the query, without being excessively verbose.

- **5**: Covers all key elements from the reference; nothing important is missing.
- **4**: Covers most key elements; one minor element missing.
- **3**: Covers some key elements; several notable omissions.
- **2**: Misses most key elements; very shallow description.
- **1**: Essentially empty or captures nothing from the reference.

### 5. Fluency and Coherence (1–5)
Is the caption well-written, grammatically correct, and logically structured?
The description should read as a natural, coherent passage rather than a disjointed
list of observations.

- **5**: Fluent, well-organized, natural prose.
- **4**: Generally well-written with minor awkwardness.
- **3**: Understandable but somewhat disjointed or repetitive.
- **2**: Poorly structured; hard to follow.
- **1**: Incoherent or ungrammatical.

## Output Format

Respond in the following JSON format ONLY. Do not include any other text.

```json
{{
  "query_relevance": {{
    "score": <1-5>,
    "rationale": "<one sentence>"
  }},
  "factual_accuracy": {{
    "score": <1-5>,
    "rationale": "<one sentence>"
  }},
  "spatial_coherence": {{
    "score": <1-5>,
    "rationale": "<one sentence>"
  }},
  "completeness": {{
    "score": <1-5>,
    "rationale": "<one sentence>"
  }},
  "fluency": {{
    "score": <1-5>,
    "rationale": "<one sentence>"
  }},
  "overall": {{
    "score": <1-5>,
    "rationale": "<one sentence summarizing the overall quality>"
  }}
}}
```
"""


# Pairwise rubric: A/B comparison, winner per dimension.
# Format fields: {query}, {reference}, {caption_a}, {caption_b}
PAIRWISE_JUDGE_PROMPT = """\
You are an expert evaluator for query-based panoramic image captioning.

## Task Description
A 360° panoramic image has been shown to two different vision-language models along with
the same user query. Both models generated a caption. Your job is to determine which
caption is better overall, considering the evaluation criteria below.

## Inputs
- **Query**: {query}
- **Reference Caption** (human-written ground truth): {reference}
- **Caption A**: {caption_a}
- **Caption B**: {caption_b}

## Evaluation Criteria (in order of importance)

1. **Spatial Coherence**: Does the caption demonstrate correct understanding of spatial
   layout in a 360° panoramic scene? Look for consistent directional language, correct
   object-to-object spatial relationships, and awareness of the wrap-around field of view.
   This is the most important criterion because it directly measures panoramic understanding.

2. **Query Relevance**: Does the caption focus on what the query asks about, rather than
   describing the entire scene?

3. **Factual Accuracy**: Are the described objects, attributes, and relationships correct
   and free from hallucination?

4. **Completeness**: Does the caption capture the key elements from the reference?

5. **Fluency**: Is the caption well-written and coherent?

## Instructions
- Compare the two captions on each criterion.
- If one caption is clearly better overall, choose it.
- Only declare a tie if the captions are genuinely comparable in quality.
- Spatial coherence should be weighted most heavily in your decision.

## Output Format

Respond in the following JSON format ONLY. Do not include any other text.

```json
{{
  "winner": "<A or B or tie>",
  "spatial_coherence": {{
    "winner": "<A or B or tie>",
    "rationale": "<one sentence>"
  }},
  "query_relevance": {{
    "winner": "<A or B or tie>",
    "rationale": "<one sentence>"
  }},
  "factual_accuracy": {{
    "winner": "<A or B or tie>",
    "rationale": "<one sentence>"
  }},
  "completeness": {{
    "winner": "<A or B or tie>",
    "rationale": "<one sentence>"
  }},
  "fluency": {{
    "winner": "<A or B or tie>",
    "rationale": "<one sentence>"
  }},
  "overall_rationale": "<two sentences explaining the overall decision>"
}}
```
"""


# Batch variant of JUDGE_PROMPT: evaluates N samples sharing one image in one API call.
# Format fields: {count}, {samples_text}
_BATCH_JUDGE_PROMPT = """\
You are an expert evaluator for query-based panoramic image captioning.

Evaluate ALL {count} sample(s) below. For each sample, compare the Candidate Caption
against the Reference Caption on five dimensions (1–5 each):

1. **Query Relevance** — Addresses the specific query?
2. **Factual Accuracy** — Objects / attributes / relations correct?
3. **Spatial Coherence** — Correct panoramic spatial understanding? (most important)
4. **Completeness** — Covers key elements from the reference?
5. **Fluency** — Well-written and grammatically correct?

{samples_text}

Return a JSON array with one object per sample (same order as above).
Do not include any other text.

```json
[
  {{
    "sample_id": <id>,
    "query_relevance":  {{"score": <1-5>, "rationale": "<one sentence>"}},
    "factual_accuracy": {{"score": <1-5>, "rationale": "<one sentence>"}},
    "spatial_coherence":{{"score": <1-5>, "rationale": "<one sentence>"}},
    "completeness":     {{"score": <1-5>, "rationale": "<one sentence>"}},
    "fluency":          {{"score": <1-5>, "rationale": "<one sentence>"}},
    "overall":          {{"score": <1-5>, "rationale": "<one sentence>"}}
  }}
]
```
"""

# Minimal system message; the domain-specific instructions live in the user prompt.
_SYSTEM_PROMPT = (
    "You are an expert evaluator for panoramic vision-language models. "
    "Return only valid JSON exactly as instructed."
)

# ---------------------------------------------------------------------------
# Dimension weights and scoring
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS: Dict[str, float] = {
    "spatial_coherence": 0.30,
    "query_relevance":   0.25,
    "factual_accuracy":  0.20,
    "completeness":      0.15,
    "fluency":           0.10,
}

# Ordered tuple of the five scored dimensions (excludes "overall")
_DIMS: Tuple[str, ...] = tuple(DIMENSION_WEIGHTS.keys())


def compute_weighted_score(scores: Dict[str, Any]) -> float:
    """Compute weighted overall score from per-dimension integer scores (1–5).

    Missing dimensions default to 3 (mid-scale).

    Returns:
        Float in [20.0, 100.0], rounded to 2 decimal places.
    """
    total = 0.0
    for dim, weight in DIMENSION_WEIGHTS.items():
        total += weight * scores.get(dim, 3)
    return round(total * 20, 2)


# ---------------------------------------------------------------------------
# Column name aliases for auto-detection
# ---------------------------------------------------------------------------

_PREDICTION_COLS = ("prediction", "output", "caption")
_REFERENCE_COLS  = ("reference", "annotation")
_QUERY_COLS      = ("query", "instruction", "original_query", "prompt")
_IMAGE_COLS      = ("image_path", "url", "file_name")

# Models that use the Responses API (GPT-5 family)
_RESPONSES_API_MODELS = frozenset({"gpt-5.2", "gpt-5.2-pro", "gpt-5-mini", "gpt-5-nano"})
_VALID_IMAGE_DETAILS = frozenset({"low", "high", "auto", "original"})

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def encode_image_base64(image_path: Path) -> Optional[str]:
    """Encode image file to base64 data-URL string."""
    if not image_path.exists() or not image_path.is_file():
        return None
    ext  = image_path.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",  ".webp": "image/webp"}.get(ext, "image/jpeg")
    b64  = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def safe_int(value: Any, default: int, low: int, high: int) -> int:
    """Clamp *value* to [low, high]; return *default* on conversion error."""
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _strip_code_fence(text: str) -> str:
    """Remove markdown ``` / ```json fences from an LLM response."""
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
    return text


# ---------------------------------------------------------------------------
# Robust JSON parsing — pointwise (nested dimension output)
# ---------------------------------------------------------------------------

# Regex fallback: match nested dimension objects and extract score / rationale
_DIM_SCORE_RE: Dict[str, re.Pattern[str]] = {
    dim: re.compile(
        rf'"{dim}"\s*:\s*\{{[^}}]*"score"\s*:\s*(\d+)',
        re.DOTALL,
    )
    for dim in (*_DIMS, "overall")
}
_DIM_RATIONALE_RE: Dict[str, re.Pattern[str]] = {
    dim: re.compile(
        rf'"{dim}"\s*:\s*\{{[^}}]*"rationale"\s*:\s*"([^"]*)',
        re.DOTALL,
    )
    for dim in (*_DIMS, "overall")
}


def _default_result(
    sample_id: Any = None,
    reason: str = "parse_or_api_failure",
) -> Dict[str, Any]:
    """Safe mid-scale defaults (score=3) that never crash downstream aggregation."""
    result: Dict[str, Any] = {"sample_id": sample_id, "model_used": ""}
    for dim in (*_DIMS, "overall"):
        result[dim]                    = 3
        result[f"{dim}_rationale"]     = reason
    result["weighted_score"] = 3.0
    return result


def _normalize_parsed(data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the nested per-dimension JSON into a flat result dict.

    Input (from judge)::

        {
            "query_relevance":  {"score": 4, "rationale": "..."},
            "factual_accuracy": {"score": 3, "rationale": "..."},
            ...
            "overall":          {"score": 4, "rationale": "..."}
        }

    Output::

        {
            "query_relevance":           4,
            "query_relevance_rationale": "...",
            ...
            "overall":                   4,
            "overall_rationale":         "...",
            "weighted_score":            3.85,
        }
    """
    result: Dict[str, Any] = {}
    for dim in (*_DIMS, "overall"):
        raw = data.get(dim, {})
        if isinstance(raw, dict):
            score     = safe_int(raw.get("score"), 3, 1, 5)
            rationale = str(raw.get("rationale", ""))
        elif isinstance(raw, (int, float)):
            # Flat (non-nested) fallback — accept bare integer
            score, rationale = safe_int(raw, 3, 1, 5), ""
        else:
            score, rationale = 3, ""
        result[dim]                = score
        result[f"{dim}_rationale"] = rationale

    result["weighted_score"] = compute_weighted_score({d: result[d] for d in _DIMS})
    return result


def _regex_extract_dims(text: str) -> Dict[str, Any]:
    """Best-effort regex extraction of dimension scores from raw text."""
    result: Dict[str, Any] = {}
    for dim in (*_DIMS, "overall"):
        m_score     = _DIM_SCORE_RE[dim].search(text)
        m_rationale = _DIM_RATIONALE_RE[dim].search(text)
        result[dim]                = safe_int(m_score.group(1), 3, 1, 5) if m_score else 3
        result[f"{dim}_rationale"] = m_rationale.group(1) if m_rationale else "regex_fallback"
    result["weighted_score"] = compute_weighted_score({d: result[d] for d in _DIMS})
    return result


def parse_judge_response(text: str) -> Dict[str, Any]:
    """Parse pointwise judge output with multiple fallback strategies.

    Order:
      1. json.loads() on cleaned text (happy path)
      2. Regex extraction of individual dimension scores
      3. Return default mid-scale scores (NEVER crash)
    """
    if not text or not text.strip():
        return _default_result(reason="empty_response")

    payload = _strip_code_fence(text.strip())

    # Strategy 1 & 2: locate and parse JSON object
    start, end = payload.find("{"), payload.rfind("}")
    if start != -1 and end > start:
        try:
            data   = json.loads(payload[start: end + 1])
            result = _normalize_parsed(data)
            return result
        except json.JSONDecodeError:
            pass

    # Strategy 3: regex extraction
    result = _regex_extract_dims(text)
    if any(result.get(d, 3) != 3 for d in _DIMS):
        return result

    # Strategy 4: hard default
    return _default_result(reason="parse_failed")


# ---------------------------------------------------------------------------
# Robust JSON parsing — pairwise
# ---------------------------------------------------------------------------

_VALID_WINNERS = {"A", "B", "TIE"}


def _canonical_winner(raw: str) -> str:
    """Normalize winner string to 'A', 'B', or 'tie'."""
    v = raw.strip().upper()
    if v == "TIE":
        return "tie"
    return v if v in ("A", "B") else "tie"


def parse_pairwise_response(text: str) -> Dict[str, Any]:
    """Parse pairwise judge output.  Returns winner + per-dimension winners."""
    if not text or not text.strip():
        return {"winner": "tie", "overall_rationale": "empty_response"}

    payload    = _strip_code_fence(text.strip())
    start, end = payload.find("{"), payload.rfind("}")

    if start != -1 and end > start:
        try:
            data: Dict[str, Any] = json.loads(payload[start: end + 1])
            result: Dict[str, Any] = {
                "winner":             _canonical_winner(str(data.get("winner", "tie"))),
                "overall_rationale":  str(data.get("overall_rationale", "")),
            }
            for dim in _DIMS:
                dim_data = data.get(dim, {})
                if isinstance(dim_data, dict):
                    result[f"{dim}_winner"]    = _canonical_winner(
                        str(dim_data.get("winner", "tie"))
                    )
                    result[f"{dim}_rationale"] = str(dim_data.get("rationale", ""))
                else:
                    result[f"{dim}_winner"]    = "tie"
                    result[f"{dim}_rationale"] = ""
            return result
        except json.JSONDecodeError:
            pass

    # Regex fallback: at minimum extract overall winner
    m = re.search(r'"winner"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    winner = _canonical_winner(m.group(1) if m else "tie")
    return {"winner": winner, "overall_rationale": "parse_fallback"}


# ---------------------------------------------------------------------------
# Robust JSON parsing — batch
# ---------------------------------------------------------------------------


def parse_batch_response(
    text: str, samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Parse a JSON-array batch response from the judge."""
    if not text or not text.strip():
        return [_default_result(s.get("sample_id"), "empty_batch") for s in samples]

    payload    = _strip_code_fence(text.strip())
    arr_start  = payload.find("[")
    arr_end    = payload.rfind("]")
    results: List[Dict[str, Any]] = []

    if arr_start != -1 and arr_end > arr_start:
        try:
            items = json.loads(payload[arr_start: arr_end + 1])
            if isinstance(items, list):
                for item in items:
                    r              = _normalize_parsed(item)
                    r["sample_id"] = item.get("sample_id")
                    results.append(r)
                return _fill_missing(results, samples)
        except json.JSONDecodeError:
            pass

    # Fallback: find individual JSON objects by sample_id
    for s in samples:
        sid = s.get("sample_id")
        pat = rf"\{{[^{{}}]*\"sample_id\"\s*:\s*{re.escape(str(sid))}[^{{}}]*\}}"
        m   = re.search(pat, text)
        if m:
            r              = parse_judge_response(m.group(0))
            r["sample_id"] = sid
        else:
            r = _default_result(sid, "batch_parse_miss")
        results.append(r)

    return results


def _fill_missing(
    results: List[Dict[str, Any]], samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Ensure every sample has a result entry."""
    seen = {r.get("sample_id") for r in results}
    for s in samples:
        sid = s.get("sample_id")
        if sid not in seen:
            results.append(_default_result(sid, "missing_in_batch"))
    return results


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def normalize_input(input_path: Path) -> pd.DataFrame:
    """Load CSV / JSON and normalise to canonical column names.

    Returns a DataFrame with columns: sample_id, query, prediction,
    reference, image_path  (plus any other original columns).
    """
    if input_path.suffix.lower() == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("predictions"), list):
            df = pd.DataFrame(data["predictions"])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("JSON must be a list or {predictions: [...]}.")
    else:
        df = pd.read_csv(input_path)

    # Auto-detect and rename to canonical columns
    renames: Dict[str, str] = {}
    for canonical, aliases in [
        ("prediction", _PREDICTION_COLS),
        ("reference",  _REFERENCE_COLS),
        ("query",      _QUERY_COLS),
        ("image_path", _IMAGE_COLS),
    ]:
        if canonical not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    renames[alias] = canonical
                    break
    if renames:
        df = df.rename(columns=renames)

    # Validate required columns
    if "prediction" not in df.columns:
        raise ValueError(
            f"No prediction column found. "
            f"Expected one of: {_PREDICTION_COLS}. Got: {list(df.columns)}"
        )
    if "reference" not in df.columns:
        raise ValueError(
            f"No reference column found. "
            f"Expected one of: {_REFERENCE_COLS}. Got: {list(df.columns)}"
        )

    # Drop rows with empty reference
    df = df[
        df["reference"].notna()
        & (df["reference"].astype(str).str.strip() != "")
    ].reset_index(drop=True)

    # Fill optional columns
    if "query"      not in df.columns: df["query"]      = ""
    if "image_path" not in df.columns: df["image_path"] = ""

    # Ensure sample_id
    if "sample_id" not in df.columns:
        for fallback in ("image_id", "id", "idx"):
            if fallback in df.columns:
                df["sample_id"] = df[fallback]
                break
        else:
            df["sample_id"] = range(len(df))

    return df


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class LLMJudge:
    """LLM-as-a-judge evaluator with dual-API support.

    Scores captions on five weighted dimensions (1–5 each):
      spatial_coherence (30%), query_relevance (25%), factual_accuracy (20%),
      completeness (15%), fluency (10%).

    Output columns added to the results DataFrame:

    ==================  =====  ========================================================
    Column              Type   Description
    ==================  =====  ========================================================
    weighted_score      float  Primary metric: DIMENSION_WEIGHTS-weighted average (1–5)
    overall             int    Judge's holistic score (1–5)
    spatial_coherence   int    Panoramic spatial understanding (1–5)
    query_relevance     int    Query focus (1–5)
    factual_accuracy    int    Fact correctness (1–5)
    completeness        int    Key element coverage (1–5)
    fluency             int    Language quality (1–5)
    *_rationale         str    One-sentence explanation for each dimension
    model_used          str    Judge model name
    ==================  =====  ========================================================
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        api_key: Optional[str] = None,
        include_image: bool = True,
        base_path: Optional[Path] = None,
        max_retries: int = 3,
        workers: int = 1,
        image_detail: str = "high",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("pip install openai") from exc

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise SystemExit("OPENAI_API_KEY required (env var or --api-key).")

        self.client        = OpenAI(api_key=resolved_key, max_retries=0)
        self.model         = model
        self.include_image = include_image
        self.base_path     = base_path or Path.cwd()
        self.max_retries   = max_retries
        self.workers       = workers
        self.image_detail  = image_detail if image_detail in _VALID_IMAGE_DETAILS else "high"
        self._use_responses = model in _RESPONSES_API_MODELS

        api_label = "Responses" if self._use_responses else "ChatCompletions"
        logger.info(
            "LLMJudge: model=%s, api=%s, image=%s, detail=%s",
            model, api_label, include_image, self.image_detail,
        )

    # -- path helpers --------------------------------------------------------

    def _resolve_image(self, raw: str) -> Optional[Path]:
        if not raw or raw == "nan":
            return None
        p = Path(raw)
        if p.is_absolute():
            return p if p.is_file() else None
        resolved = self.base_path / p
        return resolved if resolved.is_file() else None

    def _prepare_image_input(self, raw_path: str) -> Optional[str]:
        if not self.include_image:
            return None
        raw_path = raw_path.strip()
        if raw_path.startswith(("http://", "https://")):
            return raw_path
        img = self._resolve_image(raw_path)
        return encode_image_base64(img) if img else None

    # -- single pointwise evaluation -----------------------------------------

    def evaluate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single sample. Returns dict with per-dimension scores + weighted_score."""
        user_prompt = JUDGE_PROMPT.format(
            query     = sample.get("query", ""),
            reference = sample.get("reference", ""),
            candidate = sample.get("prediction", ""),
        )
        image_data_url = self._prepare_image_input(str(sample.get("image_path", "")))

        if self._use_responses:
            return self._call_responses(user_prompt, image_data_url, sample.get("sample_id"))
        return self._call_chat(user_prompt, image_data_url, sample.get("sample_id"))

    # -- single pairwise evaluation ------------------------------------------

    def evaluate_pairwise(
        self,
        query:      str,
        reference:  str,
        caption_a:  str,
        caption_b:  str,
        image_path: str = "",
        sample_id:  Any = None,
    ) -> Dict[str, Any]:
        """Compare caption A vs caption B. Returns winner + per-dimension winners."""
        user_prompt    = PAIRWISE_JUDGE_PROMPT.format(
            query=query, reference=reference, caption_a=caption_a, caption_b=caption_b,
        )
        image_data_url = self._prepare_image_input(image_path)
        raw = self._raw_api_call(user_prompt, image_data_url, max_tokens=600)
        if raw is None:
            return {"sample_id": sample_id, "winner": "tie", "overall_rationale": "api_error"}
        result              = parse_pairwise_response(raw)
        result["sample_id"] = sample_id
        result["model_used"]= self.model
        return result

    # -- batch pointwise evaluation ------------------------------------------

    def evaluate_batch(
        self,
        df:             pd.DataFrame,
        max_samples:    Optional[int] = None,
        batch_by_image: bool = False,
        partial_path:   Optional[Path] = None,
    ) -> pd.DataFrame:
        """Pointwise-score all rows in *df*.

        When *batch_by_image* is True, rows sharing the same ``image_path``
        are grouped into a single API call (significant cost saving for datasets
        with multiple query/annotation pairs per panorama).
        """
        if max_samples:
            df = df.head(max_samples).copy()

        if batch_by_image and "image_path" in df.columns:
            results = self._eval_grouped(df, partial_path=partial_path)
        else:
            results = self._eval_sequential(df)

        results_df = pd.DataFrame(results)
        overlap    = set(df.columns) & set(results_df.columns) - {"sample_id"}
        if overlap:
            results_df = results_df.drop(columns=list(overlap), errors="ignore")
        return df.merge(results_df, on="sample_id", how="left")

    # -- batch pairwise evaluation -------------------------------------------

    def evaluate_pairwise_batch(
        self,
        df_a:        pd.DataFrame,
        df_b:        pd.DataFrame,
        max_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Pairwise A/B evaluation for two prediction DataFrames.

        Both DataFrames must share the same ``sample_id`` values.
        Falls back to positional alignment if IDs don't overlap.

        Returns:
            DataFrame with winner per sample and per-dimension winners.
        """
        df_a = df_a.copy()
        df_b = df_b.copy()

        # Align by sample_id where possible
        shared_ids = list(
            pd.Index(df_a["sample_id"].tolist()).intersection(df_b["sample_id"].tolist())
        )
        if shared_ids:
            df_a = df_a.set_index("sample_id").loc[shared_ids].reset_index()
            df_b = df_b.set_index("sample_id").loc[shared_ids].reset_index()
        else:
            # Positional fallback — warn user
            logger.warning(
                "No shared sample_ids between A and B — aligning positionally. "
                "Ensure both CSVs are in the same row order."
            )
            min_len = min(len(df_a), len(df_b))
            df_a = df_a.head(min_len).reset_index(drop=True)
            df_b = df_b.head(min_len).reset_index(drop=True)

        if max_samples:
            df_a = df_a.head(max_samples)
            df_b = df_b.head(max_samples)

        results: List[Dict[str, Any]] = []
        for i in tqdm(range(len(df_a)), desc="Pairwise judge"):
            row_a = df_a.iloc[i]
            row_b = df_b.iloc[i]
            r = self.evaluate_pairwise(
                query      = str(row_a.get("query", "")),
                reference  = str(row_a.get("reference", "")),
                caption_a  = str(row_a.get("prediction", "")),
                caption_b  = str(row_b.get("prediction", "")),
                image_path = str(row_a.get("image_path", "")),
                sample_id  = row_a.get("sample_id"),
            )
            results.append(r)

        return pd.DataFrame(results)

    # -- internal sequential / grouped loops ---------------------------------

    def _eval_sequential(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM judge"):
            r               = self.evaluate(row.to_dict())
            r["model_used"] = self.model
            results.append(r)
        return results

    def _eval_grouped(self, df: pd.DataFrame, partial_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for _, row in df.iterrows():
            groups[str(row.get("image_path", ""))].append(row.to_dict())

        logger.info(
            "Batch-by-image: %d samples across %d unique images (workers=%d)",
            len(df), len(groups), self.workers,
        )

        results_map: Dict[str, List[Dict[str, Any]]] = {}
        lock = threading.Lock()
        completed_groups: List[int] = [0]
        pbar = tqdm(total=len(df), desc="LLM judge (grouped)")

        def _process(img_path: str, group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            batch_results = self._eval_image_group(img_path, group)
            for r in batch_results:
                r["model_used"] = self.model
            return batch_results

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(_process, img_path, group): img_path
                for img_path, group in groups.items()
            }
            for future in as_completed(futures):
                img_path = futures[future]
                group    = groups[img_path]
                try:
                    batch_results = future.result()
                except Exception as exc:
                    logger.error("Worker failed for %s: %s", img_path, exc)
                    batch_results = [_default_result(s.get("sample_id"), "worker_error") for s in group]
                with lock:
                    results_map[img_path] = batch_results
                    completed_groups[0] += 1
                    pbar.update(len(group))
                    if partial_path and completed_groups[0] % 25 == 0:
                        partial: List[Dict[str, Any]] = []
                        for k in list(results_map.keys()):
                            partial.extend(results_map[k])
                        pd.DataFrame(partial).to_csv(
                            partial_path, index=False, encoding="utf-8-sig",
                        )
                        logger.info(
                            "Partial save: %d groups done → %s",
                            completed_groups[0], partial_path,
                        )

        pbar.close()

        results: List[Dict[str, Any]] = []
        for img_path in groups:
            results.extend(results_map.get(img_path, []))
        return results

    def _eval_image_group(
        self, image_path: str, samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple QA pairs for one panorama in a single API call."""
        if len(samples) == 1:
            return [self.evaluate(samples[0])]

        lines: List[str] = []
        for s in samples:
            lines.append(
                f"### Sample (ID: {s.get('sample_id')})\n"
                f"- **Query**: {s.get('query', '')}\n"
                f"- **Reference**: {s.get('reference', '')}\n"
                f"- **Candidate**: {s.get('prediction', '')}\n"
            )
        batch_text     = _BATCH_JUDGE_PROMPT.format(
            count=len(samples), samples_text="\n".join(lines),
        )
        max_tokens     = min(4096, len(samples) * 400 + 300)
        image_data_url = self._prepare_image_input(image_path)
        raw            = self._raw_api_call(batch_text, image_data_url, max_tokens)

        if raw is None:
            return [_default_result(s.get("sample_id"), "batch_api_error") for s in samples]
        return parse_batch_response(raw, samples)

    # -- Batch API (async, /v1/responses) ------------------------------------

    def _resize_encode_image(self, img_path: Path, max_px: int) -> Optional[str]:
        """Encode image as base64, resizing longest edge to max_px to stay under 200 MB limit."""
        try:
            from PIL import Image as _PILImage
            import io as _io
            _PILImage.MAX_IMAGE_PIXELS = None  # trusted local files — disable bomb check
            img = _PILImage.open(img_path).convert("RGB")
            w, h = img.size
            if max(w, h) > max_px:
                scale = max_px / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), _PILImage.LANCZOS)
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/jpeg;base64,{b64}"
        except Exception as exc:
            logger.warning("Image resize failed (%s): %s — using raw encode", img_path.name, exc)
            return encode_image_base64(img_path)

    def _build_group_jsonl_request(
        self,
        custom_id: str,
        img_path: str,
        samples: List[Dict[str, Any]],
        image_max_px: int,
    ) -> Dict[str, Any]:
        """Build one Batch API JSONL request dict for an image group."""
        # -- prompt --
        if len(samples) == 1:
            s = samples[0]
            prompt_text = JUDGE_PROMPT.format(
                query=s.get("query", ""),
                reference=s.get("reference", ""),
                candidate=s.get("prediction", ""),
            )
            max_out = 800
        else:
            lines = [
                f"### Sample (ID: {s.get('sample_id')})\n"
                f"- **Query**: {s.get('query', '')}\n"
                f"- **Reference**: {s.get('reference', '')}\n"
                f"- **Candidate**: {s.get('prediction', '')}\n"
                for s in samples
            ]
            prompt_text = _BATCH_JUDGE_PROMPT.format(
                count=len(samples), samples_text="\n".join(lines),
            )
            max_out = min(4096, len(samples) * 400 + 300)

        # -- image --
        image_data_url: Optional[str] = None
        if self.include_image:
            raw_path = img_path.strip()
            if raw_path.startswith(("http://", "https://")):
                image_data_url = raw_path
            else:
                resolved = self._resolve_image(img_path)
                if resolved:
                    image_data_url = self._resize_encode_image(resolved, image_max_px)

        if self._use_responses:
            content: List[Dict[str, Any]] = []
            if image_data_url:
                content.append({
                    "type": "input_image",
                    "image_url": image_data_url,
                    "detail": self.image_detail,
                })
            content.append({"type": "input_text", "text": prompt_text})
            body = {
                "model": self.model,
                "instructions": _SYSTEM_PROMPT,
                "input": [{"role": "user", "content": content}],
                "max_output_tokens": max_out,
                "reasoning": {"effort": "medium"},
            }
        else:
            content = []
            if image_data_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data_url, "detail": self.image_detail},
                })
            content.append({"type": "text", "text": prompt_text})
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "max_tokens": max_out,
                "temperature": 0.0,
            }

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses" if self._use_responses else "/v1/chat/completions",
            "body": body,
        }

    def evaluate_via_batch_api(
        self,
        df: pd.DataFrame,
        output_path: Path,
        max_samples: Optional[int] = None,
        image_max_px: int = 1024,
    ) -> pd.DataFrame:
        """Submit all evaluations via OpenAI Batch API (async, 50% cheaper).

        Crash-safe: saves batch state to {output_stem}_batch_state.json.
        Resume: if state file exists, skips upload/submit and goes straight to polling.
        """
        import io as _io
        from collections import defaultdict

        if max_samples:
            df = df.head(max_samples).copy()

        # Group by image path
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for _, row in df.iterrows():
            groups[str(row.get("image_path", ""))].append(row.to_dict())

        logger.info(
            "Batch API: %d samples across %d images (model=%s, image_max_px=%d)",
            len(df), len(groups), self.model, image_max_px,
        )

        state_path = output_path.with_name(output_path.stem + "_batch_state.json")
        id_to_group: Dict[str, Any] = {}

        # -- Resume: load existing batch id --------------------------------
        batch_id: Optional[str] = None
        if state_path.exists():
            state = json.loads(state_path.read_text())
            batch_id = state.get("batch_id")
            id_to_group = {
                cid: (meta["img"], groups.get(meta["img"], []))
                for cid, meta in state.get("id_map", {}).items()
            }
            logger.info("Resuming existing batch: %s", batch_id)

        # -- Submit new batch -------------------------------------------
        if batch_id is None:
            requests_list: List[Dict[str, Any]] = []
            for idx, (img_path, samps) in enumerate(groups.items()):
                cid = f"img_{idx:05d}"
                id_to_group[cid] = (img_path, samps)
                requests_list.append(
                    self._build_group_jsonl_request(cid, img_path, samps, image_max_px)
                )

            jsonl_bytes = ("\n".join(json.dumps(r) for r in requests_list) + "\n").encode()
            size_mb = len(jsonl_bytes) / 1024 / 1024
            logger.info("Uploading JSONL: %d requests, %.1f MB", len(requests_list), size_mb)
            if size_mb > 195:
                raise RuntimeError(
                    f"JSONL too large ({size_mb:.1f} MB > 195 MB). "
                    f"Reduce --image-max-px (currently {image_max_px}) or split into multiple runs."
                )

            file_obj = self.client.files.create(
                file=("batch_judge.jsonl", _io.BytesIO(jsonl_bytes), "application/jsonl"),
                purpose="batch",
            )
            logger.info("File uploaded: %s", file_obj.id)

            endpoint = "/v1/responses" if self._use_responses else "/v1/chat/completions"
            batch = self.client.batches.create(
                input_file_id=file_obj.id,
                endpoint=endpoint,
                completion_window="24h",
            )
            batch_id = batch.id
            logger.info("Batch submitted: %s (status=%s)", batch_id, batch.status)

            # Save crash-recovery state
            state_path.write_text(json.dumps({
                "batch_id": batch_id,
                "input_file_id": file_obj.id,
                "id_map": {
                    cid: {"img": v[0], "n_samples": len(v[1])}
                    for cid, v in id_to_group.items()
                },
            }, indent=2))

        # -- Poll until complete ----------------------------------------
        logger.info("Polling batch %s every 60s ...", batch_id)
        while True:
            batch = self.client.batches.retrieve(batch_id)
            cnt   = batch.request_counts
            done  = getattr(cnt, "completed", 0)
            total = getattr(cnt, "total",     0)
            failed= getattr(cnt, "failed",    0)
            logger.info(
                "  status=%-15s  completed=%d/%d  failed=%d",
                batch.status, done, total, failed,
            )
            if batch.status == "completed":
                break
            if batch.status in ("failed", "expired", "cancelled"):
                raise RuntimeError(f"Batch ended with status: {batch.status}")
            time.sleep(60)

        # -- Download & parse output ------------------------------------
        output_content = self.client.files.content(batch.output_file_id).text

        all_results: List[Dict[str, Any]] = []
        for line in output_content.strip().splitlines():
            if not line.strip():
                continue
            obj       = json.loads(line)
            cid       = obj.get("custom_id", "")
            img_path, samps = id_to_group.get(cid, ("", []))
            error     = obj.get("error")
            resp_body = (obj.get("response") or {}).get("body") or {}

            if error or not resp_body:
                reason = (error or {}).get("message", "batch_error")[:60]
                for s in samps:
                    r = _default_result(s.get("sample_id"), reason)
                    r["model_used"] = self.model
                    all_results.append(r)
                continue

            # Extract text from response body
            raw_text = resp_body.get("output_text", "")
            if not raw_text and "choices" in resp_body:
                choices = resp_body.get("choices") or []
                if choices:
                    message = (choices[0] or {}).get("message") or {}
                    content = message.get("content", "")
                    if isinstance(content, str):
                        raw_text = content
                    elif isinstance(content, list):
                        parts: List[str] = []
                        for item in content:
                            if isinstance(item, dict):
                                text = item.get("text")
                                if text:
                                    parts.append(str(text))
                        raw_text = "".join(parts)
            if not raw_text:
                for out_item in resp_body.get("output", []):
                    if out_item.get("type") == "message":
                        for ci in out_item.get("content", []):
                            raw_text += ci.get("text", "") if isinstance(ci, dict) else getattr(ci, "text", "")

            if len(samps) == 1:
                result = parse_judge_response(raw_text)
                result["sample_id"] = samps[0].get("sample_id")
                result["model_used"] = self.model
                all_results.append(result)
            else:
                batch_results = parse_batch_response(raw_text, samps)
                for r in batch_results:
                    r["model_used"] = self.model
                all_results.extend(batch_results)

        # -- Build result DataFrame ------------------------------------
        results_df = pd.DataFrame(all_results)
        overlap    = set(df.columns) & set(results_df.columns) - {"sample_id"}
        if overlap:
            results_df = results_df.drop(columns=list(overlap), errors="ignore")
        merged = df.merge(results_df, on="sample_id", how="left")

        # Clean up state file on success
        if state_path.exists():
            state_path.unlink()
        logger.info("Batch API complete: %d results", len(merged))
        return merged


    # -- API dispatch --------------------------------------------------------

    def _raw_api_call(
        self,
        user_text:      str,
        image_data_url: Optional[str],
        max_tokens:     int = 800,
    ) -> Optional[str]:
        if self._use_responses:
            return self._raw_responses(user_text, image_data_url, max_tokens)
        return self._raw_chat(user_text, image_data_url, max_tokens)

    # -- Chat Completions API ------------------------------------------------

    def _call_chat(
        self,
        user_prompt:    str,
        image_data_url: Optional[str],
        sample_id:      Any,
    ) -> Dict[str, Any]:
        raw = self._raw_chat(user_prompt, image_data_url, 800)
        if raw is None:
            return _default_result(sample_id, "api_error")
        result              = parse_judge_response(raw)
        result["sample_id"] = sample_id
        return result

    def _raw_chat(
        self,
        user_text:      str,
        image_data_url: Optional[str],
        max_tokens:     int,
    ) -> Optional[str]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]
        content: List[Dict[str, Any]] = []
        if image_data_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_data_url, "detail": self.image_detail},
            })
        content.append({"type": "text", "text": user_text})
        messages.append({"role": "user", "content": content})

        last_err = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_err = str(exc)
                logger.warning(
                    "ChatCompletions attempt %d/%d: %s",
                    attempt, self.max_retries, last_err[:120],
                )
                time.sleep(1.0 * attempt)

        logger.error("ChatCompletions retries exhausted: %s", last_err[:200])
        return None

    # -- Responses API -------------------------------------------------------

    def _call_responses(
        self,
        user_prompt:    str,
        image_data_url: Optional[str],
        sample_id:      Any,
    ) -> Dict[str, Any]:
        raw = self._raw_responses(user_prompt, image_data_url, 800)
        if raw is None:
            return _default_result(sample_id, "api_error")
        result              = parse_judge_response(raw)
        result["sample_id"] = sample_id
        return result

    def _raw_responses(
        self,
        user_text:      str,
        image_data_url: Optional[str],
        max_tokens:     int,
    ) -> Optional[str]:
        content: List[Dict[str, Any]] = []
        if image_data_url:
            content.append({
                "type": "input_image",
                "image_url": image_data_url,
                "detail": self.image_detail,
            })
        content.append({"type": "input_text", "text": user_text})
        input_msgs: Any = [{"role": "user", "content": content}]

        last_err = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    instructions=_SYSTEM_PROMPT,
                    input=input_msgs,
                    reasoning={"effort": "medium"},
                    max_output_tokens=max_tokens,
                )
                if hasattr(resp, "output_text") and resp.output_text:
                    return resp.output_text
                if hasattr(resp, "output") and resp.output:
                    parts: List[str] = []
                    for item in resp.output:
                        if getattr(item, "type", None) == "message":
                            for ci in getattr(item, "content", []):
                                t = getattr(ci, "text", None)
                                if t:
                                    parts.append(t)
                    if parts:
                        return "".join(parts)
                return ""
            except Exception as exc:
                last_err = str(exc)
                is_rate_limit = "429" in last_err
                is_billing    = "quota" in last_err.lower() or "billing" in last_err.lower()
                if is_rate_limit:
                    wait = 60.0 * attempt  # 60s / 120s / 180s for rate limits
                    logger.warning(
                    "ResponsesAPI 429 attempt %d/%d — waiting %.0fs: %s",
                    attempt, self.max_retries, wait, last_err[:80],
                    )
                    time.sleep(wait)
                elif "400" in last_err:
                    logger.info("Responses 400 — falling back to ChatCompletions")
                    return self._raw_chat(user_text, image_data_url, max_tokens)
                else:
                    logger.warning(
                        "ResponsesAPI attempt %d/%d: %s",
                        attempt, self.max_retries, last_err[:120],
                    )
                    time.sleep(2.0 * attempt)
                if is_billing and attempt == 1:
                    logger.error("Billing quota exhausted — stopping retries")
                    break

        logger.error("ResponsesAPI retries exhausted: %s", last_err[:200])
        return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics from pointwise judge output DataFrame."""
    n     = len(df)
    stats: Dict[str, Any] = {"total_samples": n}

    # Weighted score — primary metric
    if "weighted_score" in df.columns:
        ws = df["weighted_score"].dropna()
        stats["evaluated_samples"]     = int(len(ws))
        stats["mean_weighted_score"]   = round(float(ws.mean()),      2) if len(ws) else 0.0
        stats["std_weighted_score"]    = round(float(ws.std(ddof=0)), 2) if len(ws) else 0.0
        stats["median_weighted_score"] = round(float(ws.median()),    2) if len(ws) else 0.0
    else:
        stats["evaluated_samples"] = n

    # Overall judge score
    if "overall" in df.columns:
        ov = df["overall"].dropna()
        if len(ov):
            stats["mean_overall"] = round(float(ov.mean()) * 20, 2)

    # Per-dimension means
    for dim in _DIMS:
        if dim in df.columns:
            vals = df[dim].dropna()
            if len(vals):
                stats[f"mean_{dim}"] = round(float(vals.mean()) * 20, 2)

    return stats


def compute_pairwise_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute win-rate statistics from pairwise judge output DataFrame."""
    n     = len(df)
    stats: Dict[str, Any] = {"total_samples": n, "evaluated_samples": n}
    if n == 0:
        return stats

    winners = df["winner"].str.upper() if "winner" in df.columns else pd.Series(dtype=str)
    stats["win_rate_a"] = float((winners == "A").sum())   / n
    stats["win_rate_b"] = float((winners == "B").sum())   / n
    stats["tie_rate"]   = float((winners == "TIE").sum()) / n

    for dim in _DIMS:
        col = f"{dim}_winner"
        if col in df.columns:
            dw = df[col].str.upper()
            stats[f"{dim}_win_rate_a"] = float((dw == "A").sum()) / n
            stats[f"{dim}_win_rate_b"] = float((dw == "B").sum()) / n

    return stats


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

_DIM_META = {
    #  dim key             display label         weight
    "spatial_coherence": ("Spatial Coherence",  "30%"),
    "query_relevance":   ("Query Relevance",    "25%"),
    "factual_accuracy":  ("Factual Accuracy",   "20%"),
    "completeness":      ("Completeness",       "15%"),
    "fluency":           ("Fluency",            "10%"),
}


def print_summary(stats: Dict[str, Any], mode: str = "pointwise") -> None:
    """Print a human-readable summary table to stdout."""
    W = 65
    print()
    print("=" * W)
    print("  LLM-as-Judge Results  —  QuIC-360 Panoramic Captioning")
    print("=" * W)
    print(
        f"  Samples   : {stats.get('evaluated_samples', 0)}"
        f"/{stats.get('total_samples', 0)}"
    )
    print(f"  Model     : {stats.get('model', 'N/A')}")
    print("-" * W)

    if mode == "pointwise":
        ws = stats.get("mean_weighted_score")
        ov = stats.get("mean_overall")
        if ws is not None:
            print(f"  Weighted Score  (primary)   : {ws:.1f} / 100")
        if ov is not None:
            print(f"  Overall Score   (judge raw) : {ov:.1f} / 100")
        print("-" * W)
        for dim, (label, weight) in _DIM_META.items():
            key = f"mean_{dim}"
            if key in stats:
                bar_w  = int(stats[key] / 100.0 * 20)
                bar    = "#" * bar_w + "-" * (20 - bar_w)
                print(f"  {label:<22s} ({weight})  {stats[key]:.1f}  |{bar}|")

    elif mode == "pairwise":
        for key, label in (
            ("win_rate_a", "Win rate  A"),
            ("win_rate_b", "Win rate  B"),
            ("tie_rate",   "Tie rate"),
        ):
            if key in stats:
                bar_w = int(stats[key] * 30)
                bar   = "#" * bar_w + "-" * (30 - bar_w)
                print(f"  {label:<14s}  {stats[key]:5.1%}  |{bar}|")
        if "win_rate_a" in stats and "win_rate_b" in stats:
            winner_label = (
                "A" if stats["win_rate_a"] > stats["win_rate_b"] else
                "B" if stats["win_rate_b"] > stats["win_rate_a"] else
                "TIE"
            )
            print(f"\n  Overall winner: {winner_label}")
        print("-" * W)
        for dim, (label, _) in _DIM_META.items():
            ka = f"{dim}_win_rate_a"
            kb = f"{dim}_win_rate_b"
            if ka in stats and kb in stats:
                d_winner = (
                    "A" if stats[ka] > stats[kb] else
                    "B" if stats[kb] > stats[ka] else
                    "tie"
                )
                print(f"  {label:<22s}  A {stats[ka]:.0%}  B {stats[kb]:.0%}  → {d_winner}")

    print("=" * W)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge for QuIC-360 panoramic VLM predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Pointwise scoring (default)\n"
            "  python scripts/llm_judge_eval.py --input predictions.csv\n"
            "  python scripts/llm_judge_eval.py --input preds.csv "
            "--model gpt-4o --batch-by-image --save-stats\n\n"
            "  # Pairwise A/B comparison\n"
            "  python scripts/llm_judge_eval.py --input model_a.csv "
            "--compare model_b.csv --model gpt-4.1-mini\n"
        ),
    )
    parser.add_argument("--input",    "-i", required=True, help="Predictions CSV or JSON (model A in pairwise mode)")
    parser.add_argument("--compare",  "-c", default=None,  help="Second CSV for pairwise A/B comparison (model B)")
    parser.add_argument("--output",   "-o", default=None,  help="Output CSV path (auto-named if omitted)")
    parser.add_argument("--model",    "-m", default="gpt-5.2", help="OpenAI model (default: gpt-5.2)")
    parser.add_argument("--api-key",  default=None,       help="OpenAI API key (reads OPENAI_API_KEY if unset)")
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate first N samples only")
    parser.add_argument("--base-path",   default=None,    help="Base directory for resolving relative image paths")
    parser.add_argument("--no-image",    action="store_true", help="Text-only evaluation (skip image encoding)")
    parser.add_argument("--batch-by-image", action="store_true",
                        help="Group samples by image for fewer API calls (~Nx cost saving)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Concurrent workers for batch-by-image grouping (default: 5)")
    parser.add_argument("--use-batch-api", action="store_true",
                        help="Use OpenAI Batch API (async, 50%% cheaper). Submits all requests at once, polls until done.")
    parser.add_argument("--image-max-px", type=int, default=1024,
                        help="Resize longest image edge to this size before base64 encoding in batch mode (default: 1024)")
    parser.add_argument("--image-detail", type=str, default="high",
                        choices=sorted(_VALID_IMAGE_DETAILS),
                        help="Vision detail level for judge image input (default: high)")
    parser.add_argument("--save-stats", action="store_true",
                        help="Save statistics JSON alongside output CSV")
    parser.add_argument("--dedup", action="store_true",
                        help="Deduplicate on (image_path, query): keep first reference per pair")
    parser.add_argument("--sample-n", type=int, default=None,
                        help="Randomly sample N rows after dedup (reproducible with --sample-seed)")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Random seed for --sample-n (default: 42)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    pairwise_mode = args.compare is not None
    if args.output:
        output_path = Path(args.output)
    else:
        suffix      = "_pairwise.csv" if pairwise_mode else "_judge_scores.csv"
        output_path = input_path.with_name(f"{input_path.stem}{suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_path) if args.base_path else input_path.parent

    logger.info("Loading input: %s", input_path)
    df = normalize_input(input_path)
    logger.info("Loaded %d samples with valid reference", len(df))

    if args.dedup:
        before = len(df)
        df = df.drop_duplicates(subset=["image_path", "query"], keep="first").reset_index(drop=True)
        df["sample_id"] = range(len(df))
        logger.info("Dedup (image_path, query): %d -> %d samples", before, len(df))

    if args.sample_n and args.sample_n < len(df):
        df = df.sample(n=args.sample_n, random_state=args.sample_seed).reset_index(drop=True)
        df["sample_id"] = range(len(df))
        logger.info("Sampled %d rows (seed=%d)", len(df), args.sample_seed)

    if len(df) == 0:
        logger.error("No valid samples found.")
        sys.exit(1)

    judge = LLMJudge(
        model         = args.model,
        api_key       = args.api_key,
        include_image = not args.no_image,
        base_path     = base_path,
        workers       = args.workers,
        image_detail  = args.image_detail,
    )

    t0 = time.time()

    if pairwise_mode:
        # ── Pairwise mode ──────────────────────────────────────────────────
        compare_path = Path(args.compare)
        if not compare_path.exists():
            logger.error("Compare file not found: %s", compare_path)
            sys.exit(1)
        df_b = normalize_input(compare_path)
        logger.info("Loaded %d samples from compare CSV (model B)", len(df_b))

        results_df = judge.evaluate_pairwise_batch(
            df, df_b, max_samples=args.max_samples,
        )
        elapsed = time.time() - t0

        results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("Saved pairwise results: %s (%.1fs)", output_path, elapsed)

        stats               = compute_pairwise_statistics(results_df)
        stats["model"]      = args.model
        stats["elapsed_seconds"] = round(elapsed, 2)
        stats["input_a"]    = str(input_path)
        stats["input_b"]    = str(compare_path)
        print_summary(stats, mode="pairwise")

    else:
        # ── Pointwise mode ─────────────────────────────────────────────────
        if args.use_batch_api:
            results_df = judge.evaluate_via_batch_api(
                df,
                output_path    = output_path,
                max_samples    = args.max_samples,
                image_max_px   = args.image_max_px,
            )
        else:
            results_df = judge.evaluate_batch(
                df,
                max_samples    = args.max_samples,
                batch_by_image = args.batch_by_image,
                partial_path   = output_path.with_name(output_path.stem + "_partial.csv"),
            )
        elapsed = time.time() - t0

        results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("Saved judge scores: %s (%.1fs)", output_path, elapsed)

        stats                     = compute_statistics(results_df)
        stats["model"]            = args.model
        stats["elapsed_seconds"]  = round(elapsed, 2)
        stats["batch_by_image"]   = args.batch_by_image
        stats["include_image"]    = not args.no_image
        print_summary(stats, mode="pointwise")

    print(f"\n  Elapsed : {elapsed:.1f}s")
    print(f"  Output  : {output_path}")

    if args.save_stats:
        stats_path = output_path.with_suffix(".stats.json")
        stats_path.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        logger.info("Saved stats: %s", stats_path)
        print(f"  Stats   : {stats_path}")


if __name__ == "__main__":
    main()
