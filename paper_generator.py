from typing import Dict, List

from question_validator import Question


def generate_question_paper(
    pool: List[Question],
    total_marks: int,
    bloom_distribution: Dict[str, float],
) -> Dict:
    """
    Constraint-based selection of questions from the validated pool.

    For now this is a simple greedy allocator that:
    - Groups by Bloom level.
    - Tries to allocate marks according to the desired distribution.
    - Fails loudly if impossible.
    """
    if total_marks <= 0:
        raise ValueError("total_marks must be positive")

    # Filter out questions without Bloom levels
    valid_pool = [q for q in pool if q.bloom_level is not None and q.marks is not None]
    
    if not valid_pool:
        raise ValueError("No questions with Bloom classification available.")

    grouped: Dict[str, List[Question]] = {}
    for q in valid_pool:
        if q.bloom_level:
            grouped.setdefault(q.bloom_level, []).append(q)

    # Compute target marks per level.
    targets: Dict[str, int] = {
        level: int(round(frac * total_marks)) for level, frac in bloom_distribution.items()
    }

    selected: List[Question] = []
    used_marks = 0
    for level, target_marks in targets.items():
        if target_marks <= 0:
            continue
        available = sorted(grouped.get(level, []), key=lambda q: q.marks or 0, reverse=True)
        level_marks = 0
        for q in available:
            if q.marks is None:
                continue
            if level_marks + q.marks > target_marks:
                continue
            selected.append(q)
            level_marks += q.marks
            used_marks += q.marks

    if used_marks == 0:
        raise ValueError("Could not allocate any marks with the given constraints.")

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


