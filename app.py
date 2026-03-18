
"""
BLOOMS BOT - Main Flask Application

Pipeline:
1. PDF Upload
2. PDF Text Extraction
3. Text Cleaning + Chunking (500-800 words)
4. GPT Question Generation (NO Bloom)
5. Question Validation (hard rejection)
6. Question Storage (SQLite)
7. GPT Bloom Classification (separate call)
8. Constraint-Based Paper Generation
9. Review & Export
"""

import json
import os
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import Flask, abort, jsonify, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

from pdf_processor import extract_text_from_pdf
from text_chunker import chunk_text
from gpt_question_gen import generate_questions_for_chunk
from question_validator import (
    build_keyword_set_from_text,
    validate_question_batch_with_report,
    Question as ValidatedQuestion,
)
from blooms_classifier import (
    classify_bloom_level_gpt,
    classify_bloom_level_heuristic,
    classify_bloom_levels_gpt_batch,
)
from paper_generator import generate_question_paper
from jobs import create_job, get_job, update_job, run_in_thread


db = SQLAlchemy()

DEFAULT_BLOOM_DISTRIBUTION = {
    "Remember": 20,
    "Understand": 30,
    "Apply": 30,
    "Analyze": 15,
    "Evaluate": 5,
    "Create": 0,
}

DEFAULT_BLOOM_PRESETS = {
    "Balanced": {
        "Remember": 20,
        "Understand": 30,
        "Apply": 30,
        "Analyze": 15,
        "Evaluate": 5,
        "Create": 0,
    },
    "Easy": {
        "Remember": 30,
        "Understand": 35,
        "Apply": 20,
        "Analyze": 10,
        "Evaluate": 5,
        "Create": 0,
    },
    "Application": {
        "Remember": 15,
        "Understand": 20,
        "Apply": 40,
        "Analyze": 20,
        "Evaluate": 5,
        "Create": 0,
    },
    "Higher Order": {
        "Remember": 10,
        "Understand": 20,
        "Apply": 25,
        "Analyze": 25,
        "Evaluate": 15,
        "Create": 5,
    },
}

DEFAULT_MARK_DISTRIBUTION = [
    {"section": "Section A", "marks_per_question": 2, "count": 10},
    {"section": "Section B", "marks_per_question": 5, "count": 4},
    {"section": "Section C", "marks_per_question": 10, "count": 1},
]

DEFAULT_DURATION_MINUTES = 90
EMAIL_REGEX = re.compile(
    r"^(?=.{6,254}$)([A-Za-z0-9](?:[A-Za-z0-9._%+-]{0,62}[A-Za-z0-9])?)@(?:[A-Za-z0-9-]{1,63}\.)+[A-Za-z]{2,63}$"
)
NAME_REGEX = re.compile(r"^[A-Za-z][A-Za-z .'-]{1,79}$")


def _utcnow() -> datetime:
    return datetime.utcnow()


class AuthUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(254), nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow)


class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(180), nullable=False)
    department = db.Column(db.String(120))
    semester = db.Column(db.String(60))
    syllabus_filename = db.Column(db.String(255))
    syllabus_pdf = db.Column(db.LargeBinary)
    created_at = db.Column(db.DateTime, default=_utcnow)

    topics = db.relationship("Topic", backref="course", cascade="all, delete-orphan", lazy=True)
    questions = db.relationship("Question", backref="course", cascade="all, delete-orphan", lazy=True)
    configs = db.relationship("PaperConfig", backref="course", cascade="all, delete-orphan", lazy=True)
    papers = db.relationship("GeneratedPaper", backref="course", cascade="all, delete-orphan", lazy=True)


class Topic(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey("course.id"), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    unit_number = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=_utcnow)

    questions = db.relationship("Question", backref="topic", cascade="all, delete-orphan", lazy=True)


class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey("course.id"), nullable=False)
    topic_id = db.Column(db.Integer, db.ForeignKey("topic.id"))
    text = db.Column(db.Text, nullable=False)
    marks = db.Column(db.Integer, nullable=False, default=0)
    bloom_level = db.Column(db.String(32))
    bloom_verb = db.Column(db.String(64))
    difficulty = db.Column(db.String(32))
    source_chunk_id = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=_utcnow)


class PaperConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey("course.id"), nullable=False)
    name = db.Column(db.String(160), nullable=False)
    total_marks = db.Column(db.Integer, nullable=False)
    duration_minutes = db.Column(db.Integer, nullable=False)
    bloom_distribution = db.Column(db.JSON, nullable=False)
    mark_distribution = db.Column(db.JSON)
    difficulty = db.Column(db.String(32))
    randomize = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=_utcnow)


class GeneratedPaper(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey("course.id"), nullable=False)
    config_id = db.Column(db.Integer, db.ForeignKey("paper_config.id"))
    title = db.Column(db.String(200), nullable=False)
    total_marks = db.Column(db.Integer, nullable=False)
    duration_minutes = db.Column(db.Integer, nullable=False)
    questions = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow)

    config = db.relationship("PaperConfig", backref="generated_papers", lazy=True)


class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    default_model_name = db.Column(db.String(120), nullable=False)
    default_difficulty = db.Column(db.String(32), nullable=False)
    default_bloom_preset = db.Column(db.String(64), nullable=False)
    default_bloom_distribution = db.Column(db.JSON, nullable=False)
    default_mark_distribution = db.Column(db.JSON, nullable=False)
    bloom_presets = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow)


def _safe_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def _normalize_email(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_valid_email(email: str) -> bool:
    return bool(EMAIL_REGEX.fullmatch(email or ""))


def _validate_password(password: str) -> Optional[str]:
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if len(password) > 64:
        return "Password must be at most 64 characters long."
    if not re.search(r"[A-Z]", password):
        return "Password must include at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must include at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Password must include at least one number."
    if not re.search(r"[^A-Za-z0-9]", password):
        return "Password must include at least one special character."
    if re.search(r"\s", password):
        return "Password cannot contain spaces."
    return None


def _is_safe_next_path(path: str) -> bool:
    if not path:
        return False
    if not path.startswith("/"):
        return False
    return not path.startswith("//")


def _resolve_next_path(default_endpoint: str) -> str:
    next_path = (request.form.get("next") or request.args.get("next") or "").strip()
    if _is_safe_next_path(next_path):
        return next_path
    return url_for(default_endpoint)


def _login_as_user(user: AuthUser) -> None:
    session.clear()
    session["auth_user_id"] = user.id
    session["auth_user_name"] = user.full_name
    session["auth_user_email"] = user.email
    session["auth_is_admin"] = False


def _login_as_admin() -> None:
    session.clear()
    session["auth_user_id"] = 0
    session["auth_user_name"] = "Administrator"
    session["auth_user_email"] = "admin"
    session["auth_is_admin"] = True


def _is_authenticated() -> bool:
    if session.get("auth_is_admin"):
        return True
    return bool(session.get("auth_user_id"))


def _difficulty_from_marks(marks: int) -> str:
    if marks <= 2:
        return "easy"
    if marks <= 5:
        return "medium"
    return "hard"


def _normalize_distribution(values: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, _safe_float(v, 0.0)) for v in values.values())
    if total <= 0:
        return {k: 0.0 for k in values}
    return {k: max(0.0, _safe_float(v, 0.0)) / total for k, v in values.items()}


def _get_bloom_distribution_raw(settings: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not settings:
        return dict(DEFAULT_BLOOM_DISTRIBUTION)
    if isinstance(settings.get("bloom_distribution"), dict):
        raw_values = settings.get("bloom_distribution")
    else:
        mapping = {
            "Remember": "bloom_remember",
            "Understand": "bloom_understand",
            "Apply": "bloom_apply",
            "Analyze": "bloom_analyze",
            "Evaluate": "bloom_evaluate",
            "Create": "bloom_create",
        }
        raw_values = {level: settings.get(key) for level, key in mapping.items()}

    values: Dict[str, float] = {}
    for level in DEFAULT_BLOOM_DISTRIBUTION:
        values[level] = max(0.0, _safe_float(raw_values.get(level), 0.0))

    if sum(values.values()) <= 0:
        return dict(DEFAULT_BLOOM_DISTRIBUTION)
    return values


def _parse_bloom_distribution(settings: Optional[Dict[str, Any]]) -> Dict[str, float]:
    raw_values = _get_bloom_distribution_raw(settings)
    if sum(raw_values.values()) <= 0:
        return _normalize_distribution(DEFAULT_BLOOM_DISTRIBUTION)
    return _normalize_distribution(raw_values)


def _resolve_total_marks(settings: Optional[Dict[str, Any]]) -> int:
    if not settings:
        return 50
    total_marks = settings.get("total_marks", 50)
    if not isinstance(total_marks, int) or total_marks <= 0:
        total_marks = _safe_int(total_marks, 50)
    if total_marks <= 0:
        return 50
    return total_marks


def _resolve_batch_sizes(settings: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    if not settings:
        return 15, 8
    batch_size = settings.get("batch_size", 15)
    retry_batch_size = settings.get("retry_batch_size", 8)
    if not isinstance(batch_size, int) or batch_size <= 0:
        batch_size = _safe_int(batch_size, 15)
    if not isinstance(retry_batch_size, int) or retry_batch_size <= 0:
        retry_batch_size = _safe_int(retry_batch_size, 8)
    return batch_size, retry_batch_size


def _run_pipeline_core(
    pdf_file,
    debug_mode: bool,
    settings: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[str, int, Optional[Dict[str, Any]]], None]] = None,
    return_bank: bool = False,
) -> Tuple[Dict[str, Any], int, Optional[List[Dict[str, Any]]]]:
    debug_report: Dict[str, Any] = {}

    def report(step: str, progress: int) -> None:
        if progress_cb:
            progress_cb(step, progress, debug_report if debug_report else None)

    report("Extracting text", 10)
    try:
        raw_text = extract_text_from_pdf(pdf_file)
    except ValueError as e:
        return {"error": str(e)}, 400, None
    except Exception as e:
        return {"error": f"Failed to process PDF: {e}"}, 500, None

    if not raw_text or not raw_text.strip():
        return {"error": "Extracted syllabus is empty; cannot generate questions."}, 400, None

    keyword_set = build_keyword_set_from_text(raw_text)

    report("Chunking text", 20)
    chunks = chunk_text(raw_text, min_words=500, max_words=800, overlap_words=100)

    if not chunks:
        return {"error": "Could not chunk syllabus text."}, 400, None

    batch_size, retry_batch_size = _resolve_batch_sizes(settings)
    debug_report.update({
        "chunks_created": len(chunks),
        "chunk_word_counts": [_approx_word_count(chunk) for chunk in chunks][:10],
        "raw_text_chars": len(raw_text),
        "raw_text_words": _approx_word_count(raw_text),
        "bloom_api_calls_count": 0,
        "bloom_batch_size": batch_size,
    })

    report("Generating questions", 35)
    all_raw_questions = []
    raw_questions_per_chunk = {}
    for chunk_id, chunk in enumerate(chunks):
        raw_questions = generate_questions_for_chunk(
            chunk_text=chunk,
            source_chunk_id=chunk_id,
        )
        raw_questions_per_chunk[chunk_id] = len(raw_questions)
        all_raw_questions.extend(raw_questions)

    if not all_raw_questions:
        return {"error": "No questions could be generated from the syllabus."}, 422, None

    debug_report["raw_questions_generated"] = len(all_raw_questions)
    debug_report["raw_questions_per_chunk"] = raw_questions_per_chunk

    question_objects = [ValidatedQuestion(q.text, q.source_chunk_id) for q in all_raw_questions]

    report("Validating questions", 50)
    valid_questions, rejected_items = validate_question_batch_with_report(
        question_objects,
        keyword_set=keyword_set,
    )

    rejection_reasons_count = {}
    rejection_examples = {}
    for item in rejected_items:
        reason = item.get("reason") or "unknown"
        rejection_reasons_count[reason] = rejection_reasons_count.get(reason, 0) + 1
        if reason not in rejection_examples:
            rejection_examples[reason] = []
        if len(rejection_examples[reason]) < 2:
            rejection_examples[reason].append(item.get("text", ""))

    debug_report["accepted_questions"] = len(valid_questions)
    debug_report["rejected_questions"] = len(rejected_items)
    debug_report["rejection_reasons_count"] = rejection_reasons_count
    debug_report["rejection_examples"] = rejection_examples

    top_rejections = sorted(
        rejection_reasons_count.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    top_rejections_text = (
        ", ".join(f"{reason}={count}" for reason, count in top_rejections)
        if top_rejections
        else "none"
    )
    print(
        f"[DEBUG] chunks={debug_report['chunks_created']}, "
        f"raw_generated={debug_report['raw_questions_generated']}, "
        f"accepted={debug_report['accepted_questions']}, "
        f"rejected={debug_report['rejected_questions']}"
    )
    print(f"[DEBUG] top_rejections: {top_rejections_text}")

    if not valid_questions:
        return {"error": "No valid questions passed validation."}, 422, None

    stored_questions = []
    for q in valid_questions:
        stored_questions.append({
            "text": q.text,
            "source_chunk_id": q.source_chunk_id,
        })

    report("Classifying Bloom levels", 70)
    classified_questions = []
    bloom_api_calls = 0
    bloom_batch_failures = 0
    bloom_retry_batches = 0
    bloom_batches_ok = 0
    bloom_batches_failed = 0
    bloom_singles_fallback = 0
    for start in range(0, len(stored_questions), batch_size):
        batch = stored_questions[start:start + batch_size]
        batch_texts = [q["text"] for q in batch]
        batch_results = classify_bloom_levels_gpt_batch(batch_texts)
        bloom_api_calls += 1

        if all(result is None for result in batch_results):
            bloom_batch_failures += 1
            retry_results = []
            for retry_start in range(0, len(batch_texts), retry_batch_size):
                retry_texts = batch_texts[retry_start:retry_start + retry_batch_size]
                retry_batch_results = classify_bloom_levels_gpt_batch(retry_texts)
                bloom_api_calls += 1
                bloom_retry_batches += 1
                retry_results.extend(retry_batch_results)

            if all(result is None for result in retry_results):
                bloom_batches_failed += 1
                for stored_q in batch:
                    classification = classify_bloom_level_heuristic(stored_q["text"])
                    if classification is None:
                        classification = classify_bloom_level_gpt(stored_q["text"])
                        bloom_api_calls += 1
                        bloom_singles_fallback += 1
                    if classification:
                        q_obj = ValidatedQuestion(
                            text=stored_q["text"],
                            source_chunk_id=stored_q["source_chunk_id"],
                        )
                        q_obj.bloom_level = classification.level
                        q_obj.bloom_verb = classification.verb
                        q_obj.marks = _assign_marks_by_bloom(classification.level)
                        classified_questions.append(q_obj)
                continue

            bloom_batches_ok += 1
            batch_results = retry_results
        else:
            bloom_batches_ok += 1

        missing_idxs = [i for i, r in enumerate(batch_results) if r is None]
        if missing_idxs:
            missing_texts = [batch_texts[i] for i in missing_idxs]

            recovered = []
            for retry_start in range(0, len(missing_texts), retry_batch_size):
                chunk = missing_texts[retry_start:retry_start + retry_batch_size]
                rb = classify_bloom_levels_gpt_batch(chunk)
                bloom_api_calls += 1
                bloom_retry_batches += 1
                recovered.extend(rb)

            for idx, rec in zip(missing_idxs, recovered):
                if rec is not None:
                    batch_results[idx] = rec

        for stored_q, classification in zip(batch, batch_results):
            if classification is None:
                classification = classify_bloom_level_heuristic(stored_q["text"])
                if classification is None:
                    classification = classify_bloom_level_gpt(stored_q["text"])
                    bloom_api_calls += 1
                    bloom_singles_fallback += 1
            if classification:
                q_obj = ValidatedQuestion(
                    text=stored_q["text"],
                    source_chunk_id=stored_q["source_chunk_id"],
                )
                q_obj.bloom_level = classification.level
                q_obj.bloom_verb = classification.verb
                q_obj.marks = _assign_marks_by_bloom(classification.level)
                classified_questions.append(q_obj)

    debug_report["bloom_classified"] = len(classified_questions)
    debug_report["bloom_failed"] = len(stored_questions) - len(classified_questions)
    debug_report["bloom_api_calls_count"] = bloom_api_calls
    debug_report["bloom_batch_size"] = batch_size
    debug_report["bloom_batch_failures"] = bloom_batch_failures
    debug_report["bloom_retry_batches"] = bloom_retry_batches
    debug_report["bank_size_total"] = len(classified_questions)
    bank_by_bloom = {}
    bank_by_marks = {}
    for q in classified_questions:
        bank_by_bloom[q.bloom_level] = bank_by_bloom.get(q.bloom_level, 0) + 1
        bank_by_marks[q.marks] = bank_by_marks.get(q.marks, 0) + 1
    debug_report["bank_size_by_bloom"] = bank_by_bloom
    debug_report["bank_size_by_marks"] = bank_by_marks

    print(
        f"[DEBUG] bloom_calls={bloom_api_calls} "
        f"bloom_batch_size={batch_size} "
        f"bloom_failed={debug_report['bloom_failed']}"
    )
    print(
        f"[DEBUG] bloom_batches_ok={bloom_batches_ok} "
        f"bloom_batches_failed={bloom_batches_failed} "
        f"bloom_singles_fallback={bloom_singles_fallback}"
    )

    if not classified_questions:
        return {"error": "No questions could be classified with Bloom levels."}, 422, None

    report("Generating paper", 85)
    total_marks = _resolve_total_marks(settings)
    bloom_distribution_raw = _get_bloom_distribution_raw(settings)
    bloom_distribution_normalized = _parse_bloom_distribution(settings)

    try:
        paper = generate_question_paper(
            pool=classified_questions,
            total_marks=total_marks,
            bloom_distribution=bloom_distribution_normalized,
        )
    except ValueError as e:
        return {"error": f"Could not generate paper: {e}"}, 422, None

    paper["bloom_distribution"] = bloom_distribution_raw

    debug_report["paper_total_marks_target"] = total_marks
    debug_report["paper_questions_selected"] = len(paper.get("questions", [])) if paper else 0

    report("Done", 100)

    bank_payload = None
    if return_bank:
        bank_payload = [
            {
                "text": q.text,
                "marks": q.marks,
                "bloom_level": q.bloom_level,
                "bloom_verb": q.bloom_verb,
                "difficulty": _difficulty_from_marks(q.marks),
                "source_chunk_id": q.source_chunk_id,
            }
            for q in classified_questions
        ]

    if debug_mode:
        return {"paper": paper, "debug": debug_report}, 200, bank_payload
    return paper, 200, bank_payload


def _assign_marks_by_bloom(bloom_level: str) -> int:
    marks_map = {
        "Remember": 2,
        "Understand": 2,
        "Apply": 5,
        "Analyze": 5,
        "Evaluate": 10,
        "Create": 10,
    }
    return marks_map.get(bloom_level, 5)


def _approx_word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _settings_to_dict(settings: Settings) -> Dict[str, Any]:
    return {
        "default_model_name": settings.default_model_name,
        "default_difficulty": settings.default_difficulty,
        "default_bloom_preset": settings.default_bloom_preset,
        "default_bloom_distribution": settings.default_bloom_distribution,
        "default_mark_distribution": settings.default_mark_distribution,
        "bloom_presets": settings.bloom_presets,
    }


def _course_to_dict(course: Course) -> Dict[str, Any]:
    return {
        "id": course.id,
        "name": course.name,
        "department": course.department,
        "semester": course.semester,
        "created_at": course.created_at.isoformat(),
        "topics_count": len(course.topics),
        "questions_count": len(course.questions),
        "has_syllabus": bool(course.syllabus_pdf),
    }


def _topic_to_dict(topic: Topic) -> Dict[str, Any]:
    return {
        "id": topic.id,
        "course_id": topic.course_id,
        "name": topic.name,
        "unit_number": topic.unit_number,
        "created_at": topic.created_at.isoformat(),
    }


def _question_to_dict(question: Question) -> Dict[str, Any]:
    return {
        "id": question.id,
        "course_id": question.course_id,
        "topic_id": question.topic_id,
        "text": question.text,
        "marks": question.marks,
        "bloom_level": question.bloom_level,
        "bloom_verb": question.bloom_verb,
        "difficulty": question.difficulty,
        "source_chunk_id": question.source_chunk_id,
        "created_at": question.created_at.isoformat(),
    }


def _config_to_dict(config: PaperConfig) -> Dict[str, Any]:
    return {
        "id": config.id,
        "course_id": config.course_id,
        "name": config.name,
        "total_marks": config.total_marks,
        "duration_minutes": config.duration_minutes,
        "bloom_distribution": config.bloom_distribution,
        "mark_distribution": config.mark_distribution,
        "difficulty": config.difficulty,
        "randomize": config.randomize,
        "created_at": config.created_at.isoformat(),
    }


def _paper_to_dict(paper: GeneratedPaper, include_questions: bool = True) -> Dict[str, Any]:
    payload = {
        "id": paper.id,
        "course_id": paper.course_id,
        "config_id": paper.config_id,
        "title": paper.title,
        "total_marks": paper.total_marks,
        "duration_minutes": paper.duration_minutes,
        "created_at": paper.created_at.isoformat(),
    }
    if include_questions:
        payload["questions"] = paper.questions
    return payload


def _get_or_create_settings() -> Settings:
    settings = Settings.query.first()
    if settings:
        return settings
    settings = Settings(
        default_model_name="gpt-4o-mini",
        default_difficulty="mixed",
        default_bloom_preset="Balanced",
        default_bloom_distribution=DEFAULT_BLOOM_DISTRIBUTION,
        default_mark_distribution=DEFAULT_MARK_DISTRIBUTION,
        bloom_presets=DEFAULT_BLOOM_PRESETS,
    )
    db.session.add(settings)
    db.session.commit()
    return settings


def init_db() -> None:
    db.create_all()
    _get_or_create_settings()


def _build_settings_from_config(config_data: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
    bloom_distribution = _parse_json_field(
        config_data.get("bloom_distribution"),
        settings.default_bloom_distribution,
    )
    mark_distribution = _parse_json_field(
        config_data.get("mark_distribution"),
        settings.default_mark_distribution,
    )
    return {
        "total_marks": _safe_int(config_data.get("total_marks"), 50),
        "duration_minutes": _safe_int(config_data.get("duration_minutes"), DEFAULT_DURATION_MINUTES),
        "bloom_distribution": bloom_distribution,
        "mark_distribution": mark_distribution,
        "difficulty": config_data.get("difficulty") or settings.default_difficulty,
        "randomize": _coerce_bool(config_data.get("randomize")),
        "batch_size": _safe_int(config_data.get("batch_size"), 15),
        "retry_batch_size": _safe_int(config_data.get("retry_batch_size"), 8),
    }


def create_app() -> Flask:
    app = Flask(__name__)

    db_path = os.path.join(app.root_path, "blooms.db")
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-this-in-production")
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    with app.app_context():
        init_db()

    @app.cli.command("init-db")
    def init_db_command() -> None:
        init_db()
        print("Initialized the database.")

    @app.route("/admin/init-db", methods=["POST", "GET"])
    def admin_init_db():
        if not app.debug:
            abort(404)
        init_db()
        return jsonify({"status": "ok"})

    @app.before_request
    def require_authentication():
        path = request.path or "/"
        if path in {"/", "/landing", "/login", "/signup", "/logout", "/favicon.ico"}:
            return None
        if path.startswith("/static/"):
            return None
        if path.startswith("/admin/init-db"):
            return None
        if _is_authenticated():
            return None
        if path.startswith("/api/"):
            return jsonify({"error": "Authentication required. Please sign in."}), 401
        next_path = path
        if request.query_string:
            next_query = request.query_string.decode("utf-8", errors="ignore")
            next_path = f"{path}?{next_query}"
        return redirect(url_for("login", next=next_path))

    @app.route("/")
    def root():
        return render_template("landing.html", page="landing", title="Blooms Bot")

    @app.route("/landing")
    def landing():
        return render_template("landing.html", page="landing", title="Blooms Bot")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            if _is_authenticated():
                return redirect(_resolve_next_path("upload"))
            info_message = None
            if request.args.get("msg") == "signed_out":
                info_message = "You have been signed out."
            return render_template(
                "login.html",
                page="login",
                title="Sign In",
                error_message=None,
                info_message=info_message,
                signup_prompt=False,
                email_value="",
                next_value=(request.args.get("next") or "").strip(),
            )

        identifier = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""
        next_value = (request.form.get("next") or "").strip()

        if identifier == "admin" and password == "123":
            _login_as_admin()
            return redirect(_resolve_next_path("upload"))

        normalized_email = _normalize_email(identifier)
        if not _is_valid_email(normalized_email):
            return render_template(
                "login.html",
                page="login",
                title="Sign In",
                error_message="Enter a valid email address. For testing, use admin / 123.",
                info_message=None,
                signup_prompt=False,
                email_value=identifier,
                next_value=next_value,
            )

        user = AuthUser.query.filter_by(email=normalized_email).first()
        if not user:
            return render_template(
                "login.html",
                page="login",
                title="Sign In",
                error_message="No account found for this email. Please sign up first.",
                info_message=None,
                signup_prompt=True,
                email_value=identifier,
                next_value=next_value,
            )

        if not check_password_hash(user.password_hash, password):
            return render_template(
                "login.html",
                page="login",
                title="Sign In",
                error_message="Incorrect password. Please try again.",
                info_message=None,
                signup_prompt=False,
                email_value=identifier,
                next_value=next_value,
            )

        _login_as_user(user)
        return redirect(_resolve_next_path("upload"))

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if request.method == "GET":
            if _is_authenticated():
                return redirect(_resolve_next_path("upload"))
            return render_template(
                "signup.html",
                page="signup",
                title="Sign Up",
                error_message=None,
                full_name_value="",
                email_value="",
                next_value=(request.args.get("next") or "").strip(),
            )

        full_name = (request.form.get("full_name") or "").strip()
        email_raw = (request.form.get("email") or "").strip()
        email_normalized = _normalize_email(email_raw)
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""
        next_value = (request.form.get("next") or "").strip()

        error_message: Optional[str] = None
        if not NAME_REGEX.fullmatch(full_name):
            error_message = "Enter a valid full name (letters, spaces, apostrophes, periods, hyphens)."
        elif not _is_valid_email(email_normalized):
            error_message = "Enter a valid email address."
        elif password != confirm_password:
            error_message = "Password and confirm password do not match."
        else:
            password_error = _validate_password(password)
            if password_error:
                error_message = password_error

        if not error_message and AuthUser.query.filter_by(email=email_normalized).first():
            error_message = "An account with this email already exists. Please sign in."

        if error_message:
            return render_template(
                "signup.html",
                page="signup",
                title="Sign Up",
                error_message=error_message,
                full_name_value=full_name,
                email_value=email_raw,
                next_value=next_value,
            )

        user = AuthUser(
            full_name=full_name,
            email=email_normalized,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return render_template(
                "signup.html",
                page="signup",
                title="Sign Up",
                error_message="An account with this email already exists. Please sign in.",
                full_name_value=full_name,
                email_value=email_raw,
                next_value=next_value,
            )

        _login_as_user(user)
        return redirect(_resolve_next_path("upload"))

    @app.route("/logout", methods=["GET"])
    def logout():
        session.clear()
        return redirect(url_for("login", msg="signed_out"))

    @app.route("/upload")
    def upload():
        return render_template("upload.html", page="upload", title="Upload Syllabus")

    @app.route("/generate")
    def generate_page():
        return render_template("generate.html", page="generate", title="Generate Paper")

    @app.route("/question-bank")
    def question_bank():
        return render_template("question_bank.html", page="question_bank", title="Question Bank")

    @app.route("/review/<int:paper_id>")
    def review(paper_id: int):
        return render_template("review.html", page="review", title="Review Paper", paper_id=paper_id)

    @app.route("/settings")
    def settings_page():
        return render_template("settings.html", page="settings", title="Settings")

    @app.route("/courses")
    def courses_page():
        return render_template("courses.html", page="courses", title="Courses")

    @app.route("/papers")
    def papers_page():
        return render_template("papers.html", page="papers", title="Generated Papers")

    def run_pipeline_job(
        job_id: str,
        pdf_bytes: Optional[bytes],
        course_id: int,
        config_id: Optional[int],
        inline_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        def report(step: str, progress: int, stats: Optional[Dict[str, Any]]) -> None:
            update_job(job_id, step=step, progress=progress, stats=stats or {})

        with app.app_context():
            update_job(job_id, step="Starting", progress=5)
            course = Course.query.get(course_id)
            if not course:
                raise RuntimeError("Course not found")

            if pdf_bytes is None:
                if not course.syllabus_pdf:
                    raise RuntimeError("No syllabus stored for this course")
                pdf_bytes = course.syllabus_pdf

            if config_id:
                config = PaperConfig.query.get(config_id)
                if not config:
                    raise RuntimeError("Config not found")
                if config.course_id != course_id:
                    raise RuntimeError("Config does not belong to this course")
                config_payload = _config_to_dict(config)
            else:
                config_payload = inline_config or {}

            settings_row = _get_or_create_settings()
            settings_payload = _build_settings_from_config(config_payload, settings_row)

            result, status_code, bank_payload = _run_pipeline_core(
                BytesIO(pdf_bytes),
                debug_mode=True,
                settings=settings_payload,
                progress_cb=report,
                return_bank=True,
            )

            if status_code != 200:
                raise RuntimeError(result.get("error", "Pipeline failed"))

            if pdf_bytes is not None:
                course.syllabus_pdf = pdf_bytes
                course.syllabus_filename = course.syllabus_filename or "syllabus.pdf"

            question_records: List[Question] = []
            for item in bank_payload or []:
                question = Question(
                    course_id=course_id,
                    topic_id=None,
                    text=item.get("text"),
                    marks=_safe_int(item.get("marks"), 0),
                    bloom_level=item.get("bloom_level"),
                    bloom_verb=item.get("bloom_verb"),
                    difficulty=item.get("difficulty"),
                    source_chunk_id=item.get("source_chunk_id"),
                )
                db.session.add(question)
                question_records.append(question)

            db.session.flush()

            lookup: Dict[Tuple[str, int, Optional[str], Optional[int]], List[int]] = {}
            for q in question_records:
                key = (
                    q.text.strip(),
                    q.marks,
                    q.bloom_level,
                    q.source_chunk_id,
                )
                lookup.setdefault(key, []).append(q.id)

            paper_payload = result.get("paper") if isinstance(result, dict) else None
            if not paper_payload:
                raise RuntimeError("Paper generation did not return a result")

            paper_questions = []
            for q in paper_payload.get("questions", []):
                key = (
                    q.get("text", "").strip(),
                    _safe_int(q.get("marks"), 0),
                    q.get("bloom_level"),
                    q.get("source_chunk_id"),
                )
                ids = lookup.get(key, [])
                question_id = ids.pop(0) if ids else None
                paper_questions.append({
                    "id": question_id,
                    "text": q.get("text"),
                    "marks": _safe_int(q.get("marks"), 0),
                    "bloom_level": q.get("bloom_level"),
                    "bloom_verb": q.get("bloom_verb"),
                    "difficulty": _difficulty_from_marks(_safe_int(q.get("marks"), 0)),
                    "source_chunk_id": q.get("source_chunk_id"),
                })

            title = config_payload.get("name") if config_payload else None
            if not title:
                title = f"Generated Paper - {course.name}"
            duration = _safe_int(config_payload.get("duration_minutes"), DEFAULT_DURATION_MINUTES)

            generated = GeneratedPaper(
                course_id=course_id,
                config_id=config_id,
                title=title,
                total_marks=_safe_int(paper_payload.get("total_marks"), settings_payload["total_marks"]),
                duration_minutes=duration,
                questions=paper_questions,
            )
            db.session.add(generated)
            db.session.commit()

            update_job(
                job_id,
                stats={
                    "stored_questions": len(question_records),
                    "paper_questions": len(paper_questions),
                },
            )

            return {
                "generated_paper_id": generated.id,
                "paper": _paper_to_dict(generated, include_questions=True),
            }

    @app.route("/api/jobs/<job_id>", methods=["GET"])
    def api_jobs(job_id: str):
        job = get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify({
            "id": job.id,
            "status": job.status,
            "progress": job.progress,
            "current_step": job.step,
            "error": job.error,
            "result": job.result,
            "debug": job.stats or {},
        })

    @app.route("/api/generate", methods=["POST"])
    def api_generate():
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            course_id = _safe_int(payload.get("course_id"), 0)
            config_id = _safe_int(payload.get("config_id"), 0) or None
            inline_config = payload.get("config") if isinstance(payload.get("config"), dict) else None
            pdf_bytes = None
        else:
            payload = request.form.to_dict()
            course_id = _safe_int(payload.get("course_id"), 0)
            config_id = _safe_int(payload.get("config_id"), 0) or None
            inline_config = _parse_json_field(payload.get("config"), None)
            pdf_file = request.files.get("syllabus_pdf")
            if pdf_file and pdf_file.filename:
                pdf_bytes = pdf_file.read()
            else:
                pdf_bytes = None

        if course_id <= 0:
            return jsonify({"error": "course_id is required"}), 400

        if not config_id and not inline_config:
            inline_config = {}

        job = create_job()
        run_in_thread(job.id, run_pipeline_job, job.id, pdf_bytes, course_id, config_id, inline_config)
        return jsonify({"job_id": job.id, "status": "queued"}), 202

    @app.route("/api/courses", methods=["GET", "POST"])
    def api_courses():
        if request.method == "GET":
            courses = Course.query.order_by(Course.created_at.desc()).all()
            return jsonify({"courses": [_course_to_dict(course) for course in courses]})

        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400
        course = Course(
            name=name,
            department=(data.get("department") or "").strip() or None,
            semester=(data.get("semester") or "").strip() or None,
        )
        db.session.add(course)
        db.session.commit()
        return jsonify({"course": _course_to_dict(course)}), 201

    @app.route("/api/courses/<int:course_id>", methods=["GET", "PUT", "DELETE"])
    def api_course_detail(course_id: int):
        course = Course.query.get_or_404(course_id)
        if request.method == "GET":
            return jsonify({"course": _course_to_dict(course)})
        if request.method == "DELETE":
            db.session.delete(course)
            db.session.commit()
            return jsonify({"status": "deleted"})

        data = request.get_json(silent=True) or {}
        if "name" in data:
            name = (data.get("name") or "").strip()
            if name:
                course.name = name
        if "department" in data:
            course.department = (data.get("department") or "").strip() or None
        if "semester" in data:
            course.semester = (data.get("semester") or "").strip() or None
        db.session.commit()
        return jsonify({"course": _course_to_dict(course)})

    @app.route("/api/courses/<int:course_id>/topics", methods=["GET", "POST"])
    def api_topics(course_id: int):
        course = Course.query.get_or_404(course_id)
        if request.method == "GET":
            topics = (
                Topic.query.filter_by(course_id=course.id)
                .order_by(Topic.unit_number.is_(None), Topic.unit_number.asc())
                .all()
            )
            return jsonify({"topics": [_topic_to_dict(topic) for topic in topics]})

        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400
        topic = Topic(
            course_id=course.id,
            name=name,
            unit_number=_safe_int(data.get("unit_number"), 0) or None,
        )
        db.session.add(topic)
        db.session.commit()
        return jsonify({"topic": _topic_to_dict(topic)}), 201

    @app.route("/api/topics/<int:topic_id>", methods=["PUT", "DELETE"])
    def api_topic_detail(topic_id: int):
        topic = Topic.query.get_or_404(topic_id)
        if request.method == "DELETE":
            db.session.delete(topic)
            db.session.commit()
            return jsonify({"status": "deleted"})
        data = request.get_json(silent=True) or {}
        if "name" in data:
            name = (data.get("name") or "").strip()
            if name:
                topic.name = name
        if "unit_number" in data:
            topic.unit_number = _safe_int(data.get("unit_number"), 0) or None
        db.session.commit()
        return jsonify({"topic": _topic_to_dict(topic)})

    @app.route("/api/questions", methods=["GET", "POST"])
    def api_questions():
        if request.method == "GET":
            query = Question.query
            course_id = request.args.get("course_id")
            topic_id = request.args.get("topic_id")
            bloom = request.args.get("bloom")
            difficulty = request.args.get("difficulty")
            marks = request.args.get("marks")
            search = request.args.get("q")

            if course_id:
                query = query.filter(Question.course_id == _safe_int(course_id, 0))
            if topic_id:
                query = query.filter(Question.topic_id == _safe_int(topic_id, 0))
            if bloom:
                query = query.filter(Question.bloom_level == bloom)
            if difficulty:
                query = query.filter(Question.difficulty == difficulty)
            if marks:
                query = query.filter(Question.marks == _safe_int(marks, 0))
            if search:
                query = query.filter(Question.text.ilike(f"%{search}%"))

            questions = query.order_by(Question.created_at.desc()).all()
            return jsonify({"questions": [_question_to_dict(q) for q in questions]})

        data = request.get_json(silent=True) or {}
        course_id = _safe_int(data.get("course_id"), 0)
        if course_id <= 0:
            return jsonify({"error": "course_id is required"}), 400
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        question = Question(
            course_id=course_id,
            topic_id=_safe_int(data.get("topic_id"), 0) or None,
            text=text,
            marks=_safe_int(data.get("marks"), 0),
            bloom_level=(data.get("bloom_level") or "").strip() or None,
            bloom_verb=(data.get("bloom_verb") or "").strip() or None,
            difficulty=(data.get("difficulty") or "").strip() or None,
            source_chunk_id=_safe_int(data.get("source_chunk_id"), 0) or None,
        )
        db.session.add(question)
        db.session.commit()
        return jsonify({"question": _question_to_dict(question)}), 201

    @app.route("/api/questions/<int:question_id>", methods=["GET", "PUT", "DELETE"])
    def api_question_detail(question_id: int):
        question = Question.query.get_or_404(question_id)
        if request.method == "GET":
            return jsonify({"question": _question_to_dict(question)})
        if request.method == "DELETE":
            db.session.delete(question)
            db.session.commit()
            return jsonify({"status": "deleted"})

        data = request.get_json(silent=True) or {}
        if "text" in data:
            text = (data.get("text") or "").strip()
            if text:
                question.text = text
        if "marks" in data:
            question.marks = _safe_int(data.get("marks"), question.marks)
        if "bloom_level" in data:
            question.bloom_level = (data.get("bloom_level") or "").strip() or None
        if "bloom_verb" in data:
            question.bloom_verb = (data.get("bloom_verb") or "").strip() or None
        if "difficulty" in data:
            question.difficulty = (data.get("difficulty") or "").strip() or None
        if "topic_id" in data:
            question.topic_id = _safe_int(data.get("topic_id"), 0) or None
        db.session.commit()
        return jsonify({"question": _question_to_dict(question)})

    @app.route("/api/configs", methods=["GET", "POST"])
    def api_configs():
        if request.method == "GET":
            course_id = request.args.get("course_id")
            query = PaperConfig.query
            if course_id:
                query = query.filter(PaperConfig.course_id == _safe_int(course_id, 0))
            configs = query.order_by(PaperConfig.created_at.desc()).all()
            return jsonify({"configs": [_config_to_dict(config) for config in configs]})

        data = request.get_json(silent=True) or {}
        course_id = _safe_int(data.get("course_id"), 0)
        if course_id <= 0:
            return jsonify({"error": "course_id is required"}), 400
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400

        config = PaperConfig(
            course_id=course_id,
            name=name,
            total_marks=_safe_int(data.get("total_marks"), 50),
            duration_minutes=_safe_int(data.get("duration_minutes"), DEFAULT_DURATION_MINUTES),
            bloom_distribution=_parse_json_field(data.get("bloom_distribution"), DEFAULT_BLOOM_DISTRIBUTION),
            mark_distribution=_parse_json_field(data.get("mark_distribution"), DEFAULT_MARK_DISTRIBUTION),
            difficulty=(data.get("difficulty") or "mixed"),
            randomize=_coerce_bool(data.get("randomize")),
        )
        db.session.add(config)
        db.session.commit()
        return jsonify({"config": _config_to_dict(config)}), 201

    @app.route("/api/configs/<int:config_id>", methods=["GET", "PUT", "DELETE"])
    def api_config_detail(config_id: int):
        config = PaperConfig.query.get_or_404(config_id)
        if request.method == "GET":
            return jsonify({"config": _config_to_dict(config)})
        if request.method == "DELETE":
            db.session.delete(config)
            db.session.commit()
            return jsonify({"status": "deleted"})

        data = request.get_json(silent=True) or {}
        if "name" in data:
            name = (data.get("name") or "").strip()
            if name:
                config.name = name
        if "total_marks" in data:
            config.total_marks = _safe_int(data.get("total_marks"), config.total_marks)
        if "duration_minutes" in data:
            config.duration_minutes = _safe_int(data.get("duration_minutes"), config.duration_minutes)
        if "bloom_distribution" in data:
            config.bloom_distribution = _parse_json_field(data.get("bloom_distribution"), config.bloom_distribution)
        if "mark_distribution" in data:
            config.mark_distribution = _parse_json_field(data.get("mark_distribution"), config.mark_distribution)
        if "difficulty" in data:
            config.difficulty = (data.get("difficulty") or "").strip() or config.difficulty
        if "randomize" in data:
            config.randomize = _coerce_bool(data.get("randomize"))
        db.session.commit()
        return jsonify({"config": _config_to_dict(config)})

    @app.route("/api/papers", methods=["GET"])
    def api_papers():
        query = GeneratedPaper.query
        course_id = request.args.get("course_id")
        if course_id:
            query = query.filter(GeneratedPaper.course_id == _safe_int(course_id, 0))
        papers = query.order_by(GeneratedPaper.created_at.desc()).all()
        return jsonify({"papers": [_paper_to_dict(p, include_questions=False) for p in papers]})

    @app.route("/api/papers/<int:paper_id>", methods=["GET"])
    def api_paper_detail(paper_id: int):
        paper = GeneratedPaper.query.get_or_404(paper_id)
        return jsonify({"paper": _paper_to_dict(paper, include_questions=True)})

    @app.route("/api/papers/<int:paper_id>/revise", methods=["POST"])
    def api_paper_revise(paper_id: int):
        paper = GeneratedPaper.query.get_or_404(paper_id)
        data = request.get_json(silent=True) or {}
        questions = data.get("questions")
        if not isinstance(questions, list) or not questions:
            return jsonify({"error": "questions list is required"}), 400

        title = (data.get("title") or f"Revised - {paper.title}").strip()
        revised = GeneratedPaper(
            course_id=paper.course_id,
            config_id=paper.config_id,
            title=title,
            total_marks=_safe_int(data.get("total_marks"), paper.total_marks),
            duration_minutes=_safe_int(data.get("duration_minutes"), paper.duration_minutes),
            questions=questions,
        )
        db.session.add(revised)
        db.session.commit()
        return jsonify({"paper": _paper_to_dict(revised, include_questions=True)}), 201

    @app.route("/api/papers/<int:paper_id>/export/pdf", methods=["POST"])
    def api_export_pdf(paper_id: int):
        _ = GeneratedPaper.query.get_or_404(paper_id)
        return jsonify({"error": "PDF export not implemented yet"}), 501

    @app.route("/api/settings", methods=["GET", "PUT"])
    def api_settings():
        settings = _get_or_create_settings()
        if request.method == "GET":
            return jsonify({"settings": _settings_to_dict(settings)})

        data = request.get_json(silent=True) or {}
        if "default_model_name" in data:
            settings.default_model_name = (data.get("default_model_name") or "").strip() or settings.default_model_name
        if "default_difficulty" in data:
            settings.default_difficulty = (data.get("default_difficulty") or "").strip() or settings.default_difficulty
        if "default_bloom_preset" in data:
            preset = (data.get("default_bloom_preset") or "").strip()
            if preset:
                settings.default_bloom_preset = preset
        if "default_bloom_distribution" in data:
            settings.default_bloom_distribution = _parse_json_field(
                data.get("default_bloom_distribution"),
                settings.default_bloom_distribution,
            )
        if "default_mark_distribution" in data:
            settings.default_mark_distribution = _parse_json_field(
                data.get("default_mark_distribution"),
                settings.default_mark_distribution,
            )
        if "bloom_presets" in data:
            settings.bloom_presets = _parse_json_field(data.get("bloom_presets"), settings.bloom_presets)
        db.session.commit()
        return jsonify({"settings": _settings_to_dict(settings)})

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
