from typing import Dict, List, Optional, Tuple

from question_validator import Question


def generate_question_paper(
    pool: List[Question],
    total_marks: int,
    bloom_distribution: Dict[str, float],
) -> Dict:
    """
    Constraint-based selection of questions from the validated pool.

    This implementation targets a 50-mark paper using mark buckets:
    - 10 questions x 2 marks = 20
    - 4 questions x 5 marks = 20
    - 1 question x 10 marks = 10
    If a bucket is short, it falls back to nearby buckets while still
    reaching the total marks.
    """
    if total_marks <= 0:
        raise ValueError("total_marks must be positive")

    # Filter out questions without Bloom levels
    valid_pool = [
        q
        for q in pool
        if q.bloom_level is not None and q.marks in {2, 5, 10}
    ]
    
    if not valid_pool:
        raise ValueError("No questions with Bloom classification available.")

    buckets = {2: [], 5: [], 10: []}
    for q in valid_pool:
        buckets[q.marks].append(q)

    desired_counts = {2: 10, 5: 4, 10: 1}
    counts = _find_best_mark_counts(
        available_counts={mark: len(items) for mark, items in buckets.items()},
        total_marks=total_marks,
        desired_counts=desired_counts,
    )
    if counts is None:
        raise ValueError("Could not allocate exact total marks with available questions.")

    selected: List[Question] = []
    for mark in (2, 5, 10):
        selected.extend(buckets[mark][: counts[mark]])

    used_marks = sum(q.marks for q in selected)
    if used_marks != total_marks:
        raise ValueError("Generated paper does not meet the total marks target.")

    paper = {
        "total_marks": used_marks,
        "questions": [
            {
                "text": q.text,
                "marks": q.marks,
                "bloom_level": q.bloom_level,
                "bloom_verb": q.bloom_verb,
                "source_chunk_id": q.source_chunk_id,
            }
            for q in selected
        ],
        "bloom_distribution": bloom_distribution,
    }
    return paper


def _find_best_mark_counts(
    available_counts: Dict[int, int],
    total_marks: int,
    desired_counts: Dict[int, int],
) -> Optional[Dict[int, int]]:
    """
    Find a feasible (2,5,10) mark count combination to hit total_marks,
    minimizing deviation from desired_counts.
    """
    avail_2 = available_counts.get(2, 0)
    avail_5 = available_counts.get(5, 0)
    avail_10 = available_counts.get(10, 0)

    best_counts: Optional[Dict[int, int]] = None
    best_score: Optional[int] = None

    max_10 = min(avail_10, total_marks // 10)
    for count_10 in range(max_10 + 1):
        remaining_after_10 = total_marks - (10 * count_10)
        max_5 = min(avail_5, remaining_after_10 // 5)
        for count_5 in range(max_5 + 1):
            remaining = remaining_after_10 - (5 * count_5)
            if remaining < 0 or remaining % 2 != 0:
                continue
            count_2 = remaining // 2
            if count_2 > avail_2:
                continue

            score = (
                abs(count_2 - desired_counts.get(2, 0))
                + abs(count_5 - desired_counts.get(5, 0))
                + abs(count_10 - desired_counts.get(10, 0))
            )
            if best_score is None or score < best_score:
                best_score = score
                best_counts = {2: count_2, 5: count_5, 10: count_10}

    return best_counts


