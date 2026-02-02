"""
Question validation with hard rejection rules.

This module enforces strict quality checks and discards bad questions.
It does NOT attempt to fix questions - bad questions are rejected.
"""

from collections import Counter
from typing import List, Optional, Set, Tuple

# Forbidden words/phrases that cause immediate rejection
FORBIDDEN_WORDS = {
    "zero",
    "unlike",
    "therefore",
    "pham",
    "something",
    "any question",
}

# Small fallback technical terms if no syllabus keywords are provided
FALLBACK_CS_TERMS = {
    "algorithm",
    "data",
    "structure",
    "database",
    "network",
    "protocol",
    "system",
    "software",
    "hardware",
    "api",
    "programming",
    "security",
}

# Basic English stopwords (lowercase)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with", "you", "your", "we", "our", "they", "their",
    "this", "these", "those", "or", "if", "then", "than", "but", "not",
    "can", "could", "should", "would", "may", "might", "do", "does", "did",
    "what", "which", "who", "whom", "why", "how", "when", "where", "so",
    "such", "about", "into", "over", "under", "between", "within", "without",
    "because", "while", "also", "there", "here", "all", "any", "some",
    "no", "yes", "one", "two", "three", "more", "most", "much", "many",
    "each", "every", "other", "another", "same", "new", "old", "use", "used",
    "using", "useful", "example", "examples", "define", "definition",
}


class Question:
    """Simple question object - Bloom classification happens later."""
    
    def __init__(self, text: str, source_chunk_id: int = None):
        self.text = text.strip()
        self.source_chunk_id = source_chunk_id
        # These will be set after Bloom classification
        self.bloom_level = None
        self.bloom_verb = None
        self.marks = None


def validate_question_batch(candidates: List[Question]) -> List[Question]:
    """
    Apply strict validation and discard any question that fails.
    
    Validation rules:
    1. Forbidden words/phrases
    2. Minimum meaningful word count (>= 6)
    3. Keyword overlap with syllabus terms or fallback CS terms
    
    Bad questions are discarded, NOT fixed.
    """
    valid, _rejected = validate_question_batch_with_report(candidates)
    return valid


def validate_question_batch_with_report(
    candidates: List[Question],
    keyword_set: Optional[Set[str]] = None,
) -> Tuple[List[Question], List[dict]]:
    """
    Apply validation and return both valid questions and rejection details.
    """
    valid = []
    rejected = []
    for q in candidates:
        is_valid, reason = _validate_question(q, keyword_set)
        if is_valid:
            valid.append(q)
        else:
            rejected.append(
                {
                    "text": q.text,
                    "source_chunk_id": q.source_chunk_id,
                    "reason": reason,
                }
            )
    return valid, rejected


def _is_valid(q: Question) -> bool:
    """Check if a question passes all validation rules."""
    is_valid, _reason = _validate_question(q, keyword_set=None)
    return is_valid


def _validate_question(
    q: Question,
    keyword_set: Optional[Set[str]],
) -> Tuple[bool, Optional[str]]:
    """Return validation result and rejection reason (if any)."""
    is_valid, reason = _passes_forbidden_filters(q)
    if not is_valid:
        return False, reason
    is_valid, reason = _has_minimum_words(q)
    if not is_valid:
        return False, reason
    is_valid, reason = _has_keyword_overlap(q, keyword_set)
    if not is_valid:
        return False, reason
    return True, None


def _passes_forbidden_filters(q: Question) -> Tuple[bool, Optional[str]]:
    """Check for forbidden words/phrases."""
    lowered = q.text.lower()
    
    # Check for forbidden words/phrases (strict)
    for forbidden in FORBIDDEN_WORDS:
        if forbidden in lowered:
            return False, f"forbidden_word:{forbidden}"

    return True, None


def _has_keyword_overlap(
    q: Question,
    keyword_set: Optional[Set[str]],
) -> Tuple[bool, Optional[str]]:
    """Check if question overlaps syllabus keywords or fallback terms."""
    tokens = _tokenize_with_acronyms(q.text)
    if keyword_set:
        if any(token in keyword_set for token in tokens):
            return True, None
    if any(token.lower() in FALLBACK_CS_TERMS for token in tokens):
        return True, None
    return False, "no_keyword_overlap"


def _has_minimum_words(q: Question) -> Tuple[bool, Optional[str]]:
    """Check if question has at least 6 meaningful words."""
    # Remove punctuation and split
    words = [w.strip(".,;:?!()[]{}") for w in q.text.split()]
    # Filter out very short words (likely not meaningful)
    meaningful_words = [w for w in words if len(w) > 2]
    if len(meaningful_words) >= 6:
        return True, None
    return False, "too_short"


def build_keyword_set_from_text(raw_text: str) -> Set[str]:
    """
    Build a keyword set from syllabus text, preserving acronyms.
    """
    if not raw_text:
        return set()

    tokens = _tokenize_with_acronyms(raw_text)
    filtered = []
    for token in tokens:
        lower = token.lower()
        if lower in STOPWORDS:
            continue
        if len(token) >= 4 or (token.isupper() and len(token) >= 2):
            filtered.append(token)

    counts = Counter(filtered)
    top_tokens = [token for token, _count in counts.most_common(200)]
    return set(top_tokens)


def _tokenize_with_acronyms(text: str) -> List[str]:
    """
    Tokenize text, lowercasing words while preserving ALL-CAPS acronyms.
    """
    tokens = []
    current = []
    for ch in text:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                token = "".join(current)
                tokens.append(_normalize_token(token))
                current = []
    if current:
        token = "".join(current)
        tokens.append(_normalize_token(token))
    return tokens


def _normalize_token(token: str) -> str:
    if _is_acronymish(token):
        return token.upper()
    return token.lower()


def _is_acronymish(token: str) -> bool:
    if len(token) < 2:
        return False
    uppercase_count = sum(1 for ch in token if ch.isupper())
    return uppercase_count >= 2
