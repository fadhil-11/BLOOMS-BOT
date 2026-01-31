from io import BytesIO
from typing import Union

from pypdf import PdfReader


def extract_text_from_pdf(file_obj: Union[BytesIO, "FileStorage"]) -> str:
    """
    Extracts and concatenates text from a PDF file-like object.

    Responsibilities:
    - Safely read uploaded syllabus/lecture notes PDFs.
    - Return a single clean text string for downstream processing.
    - Fail fast for empty / unsupported PDFs.
    """
    try:
        # `file_obj` may be a Werkzeug FileStorage; we always read bytes.
        file_obj.seek(0)
        reader = PdfReader(file_obj)
    except Exception as e:  # pragma: no cover - defensive
        raise ValueError(f"Could not read PDF: {e}")

    if not reader.pages:
        raise ValueError("PDF has no readable pages.")

    texts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:  # pragma: no cover
            page_text = ""
        texts.append(page_text)

    combined = "\n".join(texts)
    cleaned = _basic_clean(combined)
    if not cleaned.strip():
        raise ValueError("PDF text extraction produced empty content.")

    return cleaned


def _basic_clean(text: str) -> str:
    """
    Minimal cleaning:
    - Normalize whitespace.
    - Remove obviously broken artifacts.

    More aggressive cleaning (e.g. heading detection, table dropping)
    can be layered on once we inspect real syllabi.
    """
    # Collapse multiple newlines and spaces.
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]  # drop blank lines
    return "\n".join(lines)


