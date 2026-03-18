import base64
import os
from io import BytesIO
from typing import Union

from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


API_KEY_ENV = "OPENAI_API_KEY"
OCR_MODEL = os.environ.get("BLOOMSBOT_OCR_MODEL", "gpt-4o-mini")
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_MIME_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def extract_text_from_syllabus(
    file_obj: Union[BytesIO, "FileStorage"],
    filename: str = "",
    content_type: str = "",
) -> str:
    """
    Extract text from a syllabus file (PDF or image screenshot).
    """
    file_type = _detect_file_type(file_obj=file_obj, filename=filename, content_type=content_type)

    if file_type == "pdf":
        return extract_text_from_pdf(file_obj)
    if file_type == "image":
        return extract_text_from_image(file_obj=file_obj, filename=filename, content_type=content_type)

    raise ValueError(
        "Unsupported syllabus file type. Upload a PDF or an image (PNG/JPG/JPEG/WEBP/BMP/TIFF)."
    )


def extract_text_from_pdf(file_obj: Union[BytesIO, "FileStorage"]) -> str:
    """
    Extract and concatenate text from a PDF file-like object.
    """
    try:
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


def extract_text_from_image(
    file_obj: Union[BytesIO, "FileStorage"],
    filename: str = "",
    content_type: str = "",
) -> str:
    """
    Extract text from a syllabus screenshot/image using OpenAI vision.
    """
    image_bytes = _read_file_bytes(file_obj)
    if not image_bytes:
        raise ValueError("Image file is empty.")
    if len(image_bytes) > 20 * 1024 * 1024:
        raise ValueError("Image is too large. Use an image under 20 MB.")

    image_mime = _resolve_image_mime(filename=filename, content_type=content_type)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=OCR_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract text from syllabus screenshots. "
                        "Return only plain text. Preserve headings and bullet lines."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all readable syllabus text from this image. "
                                "If text is unreadable, return exactly UNREADABLE_IMAGE."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_mime};base64,{encoded}"},
                        },
                    ],
                },
            ],
        )
    except Exception as e:
        raise ValueError(f"Could not extract text from image: {e}")

    content = response.choices[0].message.content
    if isinstance(content, list):
        text = "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        text = str(content or "")

    text = text.strip()
    if text == "UNREADABLE_IMAGE":
        raise ValueError("Image text is unreadable. Please upload a clearer screenshot.")

    cleaned = _basic_clean(text)
    if not cleaned.strip():
        raise ValueError("Image text extraction produced empty content. Try a clearer screenshot.")
    return cleaned


def _get_client() -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Please add it to requirements.")
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} is not set in the environment or .env file.")
    return OpenAI(api_key=api_key)


def _detect_file_type(file_obj: Union[BytesIO, "FileStorage"], filename: str, content_type: str) -> str:
    name = (filename or "").strip().lower()
    ext = os.path.splitext(name)[1]
    ctype = (content_type or "").strip().lower()

    if ext == ".pdf" or ctype == "application/pdf":
        return "pdf"
    if ext in SUPPORTED_IMAGE_EXTENSIONS or ctype.startswith("image/"):
        return "image"

    signature = _peek_signature(file_obj)
    if signature.startswith(b"%PDF"):
        return "pdf"
    if _looks_like_image_signature(signature):
        return "image"
    return "unknown"


def _resolve_image_mime(filename: str, content_type: str) -> str:
    ctype = (content_type or "").strip().lower()
    if ctype.startswith("image/"):
        return ctype
    ext = os.path.splitext((filename or "").strip().lower())[1]
    return IMAGE_MIME_BY_EXTENSION.get(ext, "image/png")


def _read_file_bytes(file_obj: Union[BytesIO, "FileStorage"]) -> bytes:
    file_obj.seek(0)
    data = file_obj.read()
    file_obj.seek(0)
    return data


def _peek_signature(file_obj: Union[BytesIO, "FileStorage"]) -> bytes:
    file_obj.seek(0)
    signature = file_obj.read(16)
    file_obj.seek(0)
    return signature


def _looks_like_image_signature(signature: bytes) -> bool:
    if signature.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if signature.startswith(b"\xff\xd8\xff"):
        return True
    if signature.startswith(b"RIFF") and b"WEBP" in signature:
        return True
    if signature.startswith(b"BM"):
        return True
    if signature.startswith((b"II*\x00", b"MM\x00*")):
        return True
    return False


def _basic_clean(text: str) -> str:
    """
    Minimal cleaning:
    - Normalize whitespace
    - Remove blank lines
    """
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)
