"""
LLM-as-a-Judge Evaluation Prompts for QuIC-360 Panoramic Captioning.

Provides three artefacts that the evaluation pipeline consumes:

* ``JUDGE_PROMPT``          — pointwise 5-dimension rubric (1–5 each, JSON out)
* ``PAIRWISE_JUDGE_PROMPT`` — A/B comparison rubric (winner per dimension)
* ``DIMENSION_WEIGHTS``     — spatial_coherence-first weighting scheme
* ``compute_weighted_score`` — aggregates dimension scores to a single float

Usage (pointwise)::

    from cora.evaluation.judge_prompts import JUDGE_PROMPT, compute_weighted_score

    text = judge_model.generate(
        JUDGE_PROMPT.format(query=q, reference=ref, candidate=pred)
    )
    scores = parse_judge_response(text)          # returns flat dict
    ws = compute_weighted_score(scores)          # 1.0–5.0

Usage (pairwise)::

    from cora.evaluation.judge_prompts import PAIRWISE_JUDGE_PROMPT

    text = judge_model.generate(
        PAIRWISE_JUDGE_PROMPT.format(
            query=q, reference=ref, caption_a=pred_a, caption_b=pred_b
        )
    )
"""

from __future__ import annotations

from typing import Dict, Mapping, Union

# ===========================================================================
# Pointwise Judge Prompt
# ===========================================================================

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


# ===========================================================================
# Pairwise Comparison Prompt (A/B test between two models)
# ===========================================================================

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


# ===========================================================================
# Scoring Weights (for programmatic aggregation)
# ===========================================================================

DIMENSION_WEIGHTS: Dict[str, float] = {
    "spatial_coherence": 0.30,   # Most important — directly measures panoramic adaptation
    "query_relevance":   0.25,   # Core task requirement for QuIC-360
    "factual_accuracy":  0.20,   # Correctness of visual grounding
    "completeness":      0.15,   # Coverage of reference content
    "fluency":           0.10,   # Language quality (least differentiating)
}

# Ordered tuple of the five evaluation dimensions (excludes "overall")
DIMENSIONS = tuple(DIMENSION_WEIGHTS.keys())


def compute_weighted_score(scores: Mapping[str, Union[int, float]]) -> float:
    """Compute weighted overall score from per-dimension scores.

    Args:
        scores: Dict mapping dimension name to integer score (1–5).
                Missing dimensions default to 3 (mid-scale).

    Returns:
        Weighted average score (float, 1.0–5.0), rounded to 2 decimal places.

    Example::

        >>> compute_weighted_score({"spatial_coherence": 4, "query_relevance": 5,
        ...                         "factual_accuracy": 4, "completeness": 3, "fluency": 4})
        4.1
    """
    total = 0.0
    for dim, weight in DIMENSION_WEIGHTS.items():
        total += weight * scores.get(dim, 3)  # default to 3 if missing
    return round(total, 2)


# ===========================================================================
# Usage Example
# ===========================================================================

if __name__ == "__main__":
    example_scores = {
        "spatial_coherence": 4,
        "query_relevance": 5,
        "factual_accuracy": 4,
        "completeness": 3,
        "fluency": 4,
    }
    weighted = compute_weighted_score(example_scores)
    print(f"Weighted score: {weighted}")  # → 4.1

    prompt = JUDGE_PROMPT.format(
        query="What furniture is in the living room?",
        reference="The living room features a beige sofa facing a wall-mounted TV, "
                  "with a wooden coffee table in the center and two armchairs to the left.",
        candidate="There is a couch and a TV in the room with a table.",
    )
    print(prompt[:200], "...")
