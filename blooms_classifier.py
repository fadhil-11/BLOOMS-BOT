import json
import os
import re
from dataclasses import dataclass
from typing import List, Literal, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

BloomLevel = Literal["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

CLASSIFY_MODEL = os.environ.get("BLOOMSBOT_CLASSIFY_MODEL", "gpt-4o-mini")
API_KEY_ENV = "OPENAI_API_KEY"


@dataclass
class BloomClassification:
    question: str
    level: BloomLevel
    verb: str
    confidence: float


def classify_bloom_level_heuristic(question: str) -> Optional[BloomClassification]:
    """
    Lightweight heuristic classifier used as a safety net and
    for validation. We may add a GPT-based classifier on top.

    The mapping is intentionally strict: if no clear leading verb is found,
    we return None and let the validator or generator decide.
    """
    verb = _extract_leading_verb(question)
    if not verb:
        return None

    verb_lower = verb.lower()
    if verb_lower in {"define", "list", "name", "state", "identify"}:
        level: BloomLevel = "Remember"
    elif verb_lower in {"explain", "describe", "summarize", "illustrate"}:
        level = "Understand"
    elif verb_lower in {"solve", "implement", "write", "use", "apply"}:
        level = "Apply"
    elif verb_lower in {"compare", "differentiate", "analyze", "distinguish"}:
        level = "Analyze"
    elif verb_lower in {"justify", "evaluate", "critique", "assess"}:
        level = "Evaluate"
    elif verb_lower in {"design", "propose", "construct", "develop"}:
        level = "Create"
    else:
        return None

    return BloomClassification(
        question=question.strip(),
        level=level,
        verb=verb,
        confidence=0.7,  # heuristic only
    )


def _extract_leading_verb(question: str) -> Optional[str]:
    if not question:
        return None

    # Remove trailing question mark(s) and split.
    q = question.strip()
    if not q:
        return None
    words = q.split()
    if not words:
        return None

    # Often questions begin with "Q1.", "a)" etc. Strip simple prefixes.
    first = words[0].rstrip(".:)")
    if first.lower().startswith("q") and first[1:].isdigit():
        if len(words) >= 2:
            return words[1].rstrip(".:)")
        return None

    return first


def _get_client() -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Please add it to requirements.")
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} is not set in the environment or .env file.")
    return OpenAI(api_key=api_key)


def classify_bloom_level_gpt(question: str) -> Optional[BloomClassification]:
    """
    Classify a question's Bloom's Taxonomy level using GPT.
    
    This is a SEPARATE GPT call that happens AFTER question generation.
    It does NOT rewrite the question, only classifies it.
    
    Args:
        question: The question text to classify
    
    Returns:
        BloomClassification with level, verb, and confidence, or None if classification fails
    """
    if not question or not question.strip():
        return None
    
    client = _get_client()
    
    prompt = f"""Classify the following Computer Science exam question according to Bloom's Taxonomy.

Question: {question}

Return ONLY one of the following levels:
Remember, Understand, Apply, Analyze, Evaluate, Create

Do not include any explanation or additional text."""

    try:
        response = client.chat.completions.create(
            model=CLASSIFY_MODEL,
            temperature=0.1,  # Low temperature for consistent classification
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        raw_output = response.choices[0].message.content
        if not raw_output:
            return None

        level_str = _parse_level(raw_output)
        if level_str is None:
            return None

        verb = _extract_leading_verb(question) or ""
        return BloomClassification(
            question=question.strip(),
            level=level_str,
            verb=verb,
            confidence=0.9,
        )
    
    except Exception as e:
        print(f"Error classifying question with GPT: {e}")
        return None


def _parse_level(text: str) -> Optional[BloomLevel]:
    if not text:
        return None
    for level in ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"):
        if re.search(rf"\\b{level}\\b", text, re.IGNORECASE):
            return level  # type: ignore
    return None


def _parse_batch_levels(json_text: str, n: int) -> List[Optional[BloomLevel]]:
    if not json_text or n <= 0:
        return [None] * max(n, 0)

    try:
        payload = json.loads(json_text)
    except Exception:
        return [None] * n

    if not isinstance(payload, list) or len(payload) != n:
        return [None] * n

    allowed = {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}
    results: List[Optional[BloomLevel]] = [None] * n
    seen = set()

    for item in payload:
        if not isinstance(item, dict):
            return [None] * n
        i = item.get("i")
        level = item.get("level")
        if not isinstance(i, int) or i < 1 or i > n or i in seen:
            return [None] * n
        if level not in allowed:
            return [None] * n
        results[i - 1] = level  # type: ignore
        seen.add(i)

    if any(level is None for level in results):
        return [None] * n

    return results


def classify_bloom_levels_gpt_batch(
    questions: List[str],
) -> List[Optional[BloomClassification]]:
    """
    Classify multiple questions in a single GPT call using strict JSON output.
    """
    if not questions:
        return []

    numbered = "\n".join(
        f"Q{i + 1}. {q.strip()}" for i, q in enumerate(questions)
    )
    prompt = (
        "Classify each question according to Bloom's Taxonomy.\n\n"
        "Input:\n"
        f"{numbered}\n\n"
        "Output: Return ONLY a strict JSON array of objects with keys "
        '"i" and "level", e.g. '
        '[{"i":1,"level":"Remember"},{"i":2,"level":"Apply"}]. '
        "No extra text."
    )

    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=CLASSIFY_MODEL,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        raw_output = response.choices[0].message.content or ""
        levels = _parse_batch_levels(raw_output, len(questions))
    except Exception as e:
        print(f"Error classifying question batch with GPT: {e}")
        return [None] * len(questions)

    results: List[Optional[BloomClassification]] = []
    for question, level in zip(questions, levels):
        if level is None:
            results.append(None)
            continue
        verb = _extract_leading_verb(question) or ""
        results.append(
            BloomClassification(
                question=question.strip(),
                level=level,
                verb=verb,
                confidence=0.9,
            )
        )

    return results


def classify_questions_batch(questions: List[str]) -> List[Optional[BloomClassification]]:
    """
    Classify multiple questions using GPT.
    
    Args:
        questions: List of question texts
    
    Returns:
        List of BloomClassification objects (or None for failed classifications)
    """
    results = []
    for question in questions:
        classification = classify_bloom_level_gpt(question)
        results.append(classification)
    return results


