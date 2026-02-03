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
ALLOWED_BLOOM_LEVELS = ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create")
ALLOWED_BLOOM_LEVELS_MAP = {level.lower(): level for level in ALLOWED_BLOOM_LEVELS}
WH_WORDS = {"who", "what", "when", "where", "why", "how"}
BLOOM_SYNONYMS = {
    "remembering": "Remember",
    "knowledge": "Remember",
    "understanding": "Understand",
    "comprehension": "Understand",
    "applying": "Apply",
    "application": "Apply",
    "analyzing": "Analyze",
    "analysis": "Analyze",
    "evaluating": "Evaluate",
    "evaluation": "Evaluate",
    "creating": "Create",
    "creation": "Create",
    "synthesis": "Create",
}


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
    for level in ALLOWED_BLOOM_LEVELS:
        if re.search(rf"\\b{level}\\b", text, re.IGNORECASE):
            return level  # type: ignore
    return None


def _normalize_bloom_level(value: object) -> Optional[BloomLevel]:
    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    if key in ALLOWED_BLOOM_LEVELS_MAP:
        return ALLOWED_BLOOM_LEVELS_MAP[key]  # type: ignore
    if key in BLOOM_SYNONYMS:
        return BLOOM_SYNONYMS[key]  # type: ignore
    return None


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\\s*```$", "", cleaned)
    return cleaned.strip()


def _repair_json(text: str) -> str:
    # conservative trailing-comma removal
    return re.sub(r",\\s*([}\\]])", r"\\1", text)


def _extract_json_any(text: str) -> Optional[str]:
    if not text:
        return None

    cleaned = _strip_code_fences(text)

    # If it's already a pure JSON array/object, keep it.
    if (cleaned.startswith("[") and cleaned.endswith("]")) or (
        cleaned.startswith("{") and cleaned.endswith("}")
    ):
        return cleaned

    # Otherwise slice out the largest JSON-looking region.
    first_arr, last_arr = cleaned.find("["), cleaned.rfind("]")
    first_obj, last_obj = cleaned.find("{"), cleaned.rfind("}")

    candidates = []
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        candidates.append(cleaned[first_arr : last_arr + 1])
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        candidates.append(cleaned[first_obj : last_obj + 1])

    if not candidates:
        return None

    # Prefer arrays because that's what you prompt for.
    candidates.sort(key=lambda s: 0 if s.lstrip().startswith("[") else 1)
    return candidates[0].strip()


def _parse_batch_levels(raw_text: str, n: int) -> List[Optional[BloomLevel]]:
    if not raw_text or n <= 0:
        return [None] * max(n, 0)

    extracted = _extract_json_any(raw_text)
    if not extracted:
        return [None] * n

    extracted = _repair_json(extracted)

    try:
        payload = json.loads(extracted)
    except Exception:
        return [None] * n

    out: List[Optional[BloomLevel]] = [None] * n

    # --- Case 1: payload is a list ---
    if isinstance(payload, list):
        # 1A) list of dicts (preferred): [{"i":1,"level":"Remember"}, ...]
        dict_items = [x for x in payload if isinstance(x, dict)]
        if dict_items:
            levels_by_i = {}

            for item in dict_items:
                i = item.get("i")
                level = _normalize_bloom_level(item.get("level"))

                # accept i as int, or numeric string like "3"
                if isinstance(i, str) and i.isdigit():
                    i = int(i)

                if isinstance(i, int) and 1 <= i <= n and level:
                    if i not in levels_by_i:
                        levels_by_i[i] = level

            # fill indexed matches
            for i, lvl in levels_by_i.items():
                out[i - 1] = lvl

            # 1B) positional salvage if still missing and payload length == n
            if any(x is None for x in out) and len(payload) == n:
                for idx, item in enumerate(payload[:n]):
                    if out[idx] is not None:
                        continue

                    if isinstance(item, dict):
                        lvl = _normalize_bloom_level(item.get("level"))
                        if lvl:
                            out[idx] = lvl
                    elif isinstance(item, str):
                        lvl = _normalize_bloom_level(item)
                        if lvl:
                            out[idx] = lvl

            return out

        # 1C) list of strings: ["Remember","Apply",...]
        if all(isinstance(x, str) for x in payload):
            for idx, s in enumerate(payload[:n]):
                out[idx] = _normalize_bloom_level(s)
            return out

        return out

    # --- Case 2: payload is a dict ---
    if isinstance(payload, dict):
        # 2A) {"items":[...]}
        items = payload.get("items")
        if isinstance(items, list):
            return _parse_batch_levels(json.dumps(items), n)

        # 2B) {"1":"Remember","2":"Apply"} mapping
        for k, v in payload.items():
            try:
                i = int(k)
            except Exception:
                continue
            if 1 <= i <= n:
                out[i - 1] = _normalize_bloom_level(v)

        return out

    return out


def classify_bloom_levels_gpt_batch(
    questions: List[str],
) -> List[Optional[BloomClassification]]:
    """
    Classify multiple questions in a single GPT call using strict JSON output.
    """
    if not questions:
        return []

    allowed_str = ", ".join(ALLOWED_BLOOM_LEVELS)
    numbered = "\n".join(
        f"{i + 1}. {q.strip()}" for i, q in enumerate(questions)
    )
    prompt = (
        "Classify each question according to Bloom's Taxonomy.\n\n"
        "Questions:\n"
        f"{numbered}\n\n"
        "Return ONLY a JSON array. No code fences. No extra text.\n"
        f"Each item must be: {{\"i\": <int>, \"level\": "
        f"\"<one of {allowed_str}>\"}}.\n"
        "Example for N=2:\n"
        "[{\"i\":1,\"level\":\"Remember\"},{\"i\":2,\"level\":\"Apply\"}]"
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
        missing = sum(1 for x in levels if x is None)
        if missing:
            print(f"[BloomBatch] Salvaged {len(levels) - missing}/{len(levels)} (missing={missing})")
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


