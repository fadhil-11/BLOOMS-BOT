import json
import os
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

Return ONLY a JSON object with this exact structure:
{{
  "level": "<one of: Remember, Understand, Apply, Analyze, Evaluate, Create>",
  "verb": "<the main cognitive verb in the question>",
  "confidence": <float between 0.0 and 1.0>
}}

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
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content
        if not raw_output:
            return None

        parsed = json.loads(raw_output)
        
        level_str = parsed.get("level", "").strip()
        if level_str not in {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}:
            return None
        
        verb = parsed.get("verb", "").strip()
        confidence = float(parsed.get("confidence", 0.5))
        
        return BloomClassification(
            question=question.strip(),
            level=level_str,  # type: ignore
            verb=verb,
            confidence=confidence,
        )
    
    except Exception as e:
        print(f"Error classifying question with GPT: {e}")
        return None


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


