"""
Question validation with hard rejection rules.

This module enforces strict quality checks and discards bad questions.
It does NOT attempt to fix questions - bad questions are rejected.
"""

from typing import List

# Forbidden words/phrases that cause immediate rejection
FORBIDDEN_WORDS = {
    "zero",
    "unlike",
    "therefore",
    "pham",
    "something",
    "any question",
}

# Forbidden standalone words (must appear as complete words, not substrings)
FORBIDDEN_STANDALONE = {
    "zero",
    "unlike",
    "therefore",
}

# Technical nouns that should appear in CS questions
# This is a minimal set - questions should reference real CS concepts
CS_TECHNICAL_TERMS = {
    "algorithm", "algorithms",
    "data structure", "data structures",
    "array", "arrays",
    "linked list", "linked lists",
    "stack", "stacks",
    "queue", "queues",
    "tree", "trees",
    "graph", "graphs",
    "hash", "hashing",
    "sorting", "sort",
    "searching", "search",
    "complexity", "time complexity", "space complexity",
    "operating system", "operating systems",
    "process", "processes",
    "thread", "threads",
    "scheduling",
    "deadlock",
    "database", "databases",
    "sql",
    "transaction", "transactions",
    "normalization",
    "network", "networking",
    "protocol", "protocols",
    "tcp", "udp",
    "http", "https",
    "routing",
    "encryption",
    "compiler", "compilers",
    "parsing",
    "automata",
    "finite automaton",
    "turing machine",
    "machine learning",
    "neural network",
    "programming",
    "code",
    "function", "functions",
    "class", "classes",
    "object", "objects",
    "variable", "variables",
    "loop", "loops",
    "recursion",
    "pointer", "pointers",
    "memory",
    "file", "files",
    "interface", "interfaces",
    "api",
    "software",
    "hardware",
    "system",
    "architecture",
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
    1. Basic format (length, question mark)
    2. Forbidden words/phrases
    3. Technical noun presence
    4. Minimum meaningful word count
    
    Bad questions are discarded, NOT fixed.
    """
    valid = []
    for q in candidates:
        if _is_valid(q):
            valid.append(q)
    return valid


def _is_valid(q: Question) -> bool:
    """Check if a question passes all validation rules."""
    if not _is_basic_format_valid(q):
        return False
    if not _passes_forbidden_filters(q):
        return False
    if not _has_technical_noun(q):
        return False
    if not _has_minimum_words(q):
        return False
    return True


def _is_basic_format_valid(q: Question) -> bool:
    """Check basic format: non-empty, has question mark, minimum length."""
    txt = q.text.strip()
    if not txt:
        return False
    if "?" not in txt:
        return False
    if len(txt) < 20:  # Minimum reasonable question length
        return False
    return True


def _passes_forbidden_filters(q: Question) -> bool:
    """Check for forbidden words/phrases."""
    lowered = q.text.lower()
    
    # Check for forbidden phrases
    for forbidden in FORBIDDEN_WORDS:
        if forbidden in lowered:
            return False
    
    # Check for forbidden standalone words (as complete words)
    words = {w.strip(".,;:?!()[]{}") for w in lowered.split()}
    for forbidden_word in FORBIDDEN_STANDALONE:
        if forbidden_word in words:
            # Allow "zero" only if it's clearly numerical (e.g., "zero-based indexing" is OK)
            if forbidden_word == "zero":
                # Check if it's part of a technical term
                if "zero-based" in lowered or "zero-indexed" in lowered:
                    continue
                # Check if there's a number nearby
                if any(char.isdigit() for char in lowered):
                    continue
            return False
    
    # Reject vague phrases
    if "any question" in lowered:
        return False
    if "give an example" in lowered and not any(term in lowered for term in CS_TECHNICAL_TERMS):
        return False
    
    # Reject "Professor" unless it's clearly part of a topic name
    if "professor" in words:
        # Allow if it's part of a known pattern (e.g., "Professor's algorithm")
        if "professor" in lowered and not any(
            pattern in lowered for pattern in ["algorithm", "method", "approach", "technique"]
        ):
            return False
    
    return True


def _has_technical_noun(q: Question) -> bool:
    """Check if question contains at least one technical CS term."""
    lowered = q.text.lower()
    
    # Check for technical terms (case-insensitive)
    for term in CS_TECHNICAL_TERMS:
        if term.lower() in lowered:
            return True
    
    return False


def _has_minimum_words(q: Question) -> bool:
    """Check if question has at least 6 meaningful words."""
    # Remove punctuation and split
    words = [w.strip(".,;:?!()[]{}") for w in q.text.split()]
    # Filter out very short words (likely not meaningful)
    meaningful_words = [w for w in words if len(w) > 2]
    return len(meaningful_words) >= 6
