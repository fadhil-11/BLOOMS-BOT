"""
GPT-powered question generation for BLOOMS BOT.

STRICT RULES:
- GPT generates questions ONLY from syllabus content
- NO Bloom's Taxonomy in question generation
- One API call = question generation only
- Questions must be validated before use
"""

import os
import re
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


GEN_MODEL = os.environ.get("BLOOMSBOT_GEN_MODEL", "gpt-4o-mini")
API_KEY_ENV = "OPENAI_API_KEY"


class GeneratedQuestion:
    """Simple question object without Bloom classification."""
    
    def __init__(self, text: str, source_chunk_id: Optional[int] = None):
        self.text = text.strip()
        self.source_chunk_id = source_chunk_id


def _get_client() -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Please add it to requirements.")
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} is not set in the environment or .env file.")
    return OpenAI(api_key=api_key)


# EXACT PROMPT AS SPECIFIED
QUESTION_GENERATION_PROMPT = """You are a university-level Computer Science examiner responsible for setting internal and end-semester examination papers.

Generate clear, meaningful, exam-ready questions strictly based on the syllabus content provided.

STRICT RULES:
- Every question must test a specific Computer Science concept explicitly mentioned in the syllabus.
- Every technical noun used must appear in the syllabus or be a directly related standard CS term.
- Do NOT invent abstract or placeholder topic names.
- Do NOT use vague words such as Zero, Unlike, Therefore, Something, Any Question.
- Questions must be suitable for a real written university exam.

QUESTION REQUIREMENTS:
- Generate exactly 12 questions.
- Mix of short-answer and descriptive style questions.
- Use appropriate academic verbs (define, explain, differentiate, write, implement).

SYLLABUS CONTENT:
<<<SYLLABUS_TEXT_CHUNK>>>

OUTPUT FORMAT (STRICT):
Q1. <question>
Q2. <question>
Q3. <question>
Q4. <question>
Q5. <question>
Q6. <question>
Q7. <question>
Q8. <question>
Q9. <question>
Q10. <question>
Q11. <question>
Q12. <question>

Do NOT include Bloom levels, difficulty, marks, explanations, or extra text."""


def generate_questions_for_chunk(
    chunk_text: str,
    source_chunk_id: Optional[int] = None,
    temperature: float = 0.3,
) -> List[GeneratedQuestion]:
    """
    Generate questions from a syllabus chunk using GPT.
    
    This function:
    - Does NOT include Bloom's Taxonomy
    - Does NOT validate questions (validation happens separately)
    - Returns raw questions that must be validated
    
    Args:
        chunk_text: The syllabus text chunk to generate questions from
        source_chunk_id: Optional identifier for the source chunk
        temperature: GPT temperature (default 0.3 for consistency)
    
    Returns:
        List of GeneratedQuestion objects (text only, no Bloom levels)
    """
    if not chunk_text or not chunk_text.strip():
        return []

    client = _get_client()
    
    # Replace the placeholder in the prompt with actual syllabus text
    prompt = QUESTION_GENERATION_PROMPT.replace("<<<SYLLABUS_TEXT_CHUNK>>>", chunk_text)

    try:
        response = client.chat.completions.create(
            model=GEN_MODEL,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        raw_output = response.choices[0].message.content
        if not raw_output:
            return []

        # Parse the output format: Q1. <question>\nQ2. <question>...
        questions = _parse_question_output(raw_output, source_chunk_id)
        return questions

    except Exception as e:
        # Fail silently for individual chunks - let validation handle empty results
        print(f"Error generating questions for chunk {source_chunk_id}: {e}")
        return []


def _parse_question_output(output: str, source_chunk_id: Optional[int]) -> List[GeneratedQuestion]:
    """
    Parse the GPT output in the format:
    Q1. <question>
    Q2. <question>
    ...
    """
    questions = []
    
    # Split by lines and look for Q1., Q2., etc.
    lines = output.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Match pattern: Q1. <question> or Q1 <question>
        match = re.match(r'^Q\d+\.?\s+(.+)$', line, re.IGNORECASE)
        if match:
            question_text = match.group(1).strip()
            if question_text and len(question_text) > 5:
                questions.append(GeneratedQuestion(
                    text=question_text,
                    source_chunk_id=source_chunk_id
                ))
    
    # If regex didn't work, try to extract questions by looking for numbered items
    if not questions:
        # Try alternative pattern: look for numbered questions
        for line in lines:
            line = line.strip()
            # Look for lines starting with numbers or Q followed by text
            if re.match(r'^(Q?\d+[\.\)]\s+|Q?\d+\s+)', line, re.IGNORECASE):
                # Extract everything after the number/Q prefix
                question_text = re.sub(r'^(Q?\d+[\.\)]\s+|Q?\d+\s+)', '', line, flags=re.IGNORECASE).strip()
                if question_text and len(question_text) > 5:
                    questions.append(GeneratedQuestion(
                        text=question_text,
                        source_chunk_id=source_chunk_id
                    ))
    
    return questions
