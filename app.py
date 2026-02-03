"""
BLOOMS BOT - Main Flask Application

Pipeline:
1. PDF Upload
2. PDF Text Extraction
3. Text Cleaning + Chunking (500-800 words)
4. GPT Question Generation (NO Bloom)
5. Question Validation (hard rejection)
6. Question Storage (JSON)
7. GPT Bloom Classification (separate call)
8. Constraint-Based Paper Generation
9. Review & Export
"""

import json
import os
from flask import Flask, render_template, request, jsonify

from pdf_processor import extract_text_from_pdf
from text_chunker import chunk_text
from gpt_question_gen import generate_questions_for_chunk, GeneratedQuestion
from question_validator import (
    build_keyword_set_from_text,
    validate_question_batch_with_report,
    Question,
)
from blooms_classifier import (
    classify_bloom_level_gpt,
    classify_bloom_level_heuristic,
    classify_bloom_levels_gpt_batch,
)
from paper_generator import generate_question_paper


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)
    
    # Storage for questions (in-memory for now, can be replaced with SQLite)
    questions_storage = []

    @app.route("/", methods=["GET"])
    def index():
        """Upload and configuration page."""
        return render_template("index.html")

    @app.route("/api/generate", methods=["POST"])
    def api_generate():
        """
        Main generation endpoint following the required pipeline.
        """
        debug_mode = request.args.get("debug") == "1"
        # Step 1: PDF Upload (handled by Flask)
        if "syllabus_pdf" not in request.files:
            return jsonify({"error": "No PDF uploaded"}), 400

        pdf_file = request.files["syllabus_pdf"]
        if pdf_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        try:
            # Step 2: PDF Text Extraction
            raw_text = extract_text_from_pdf(pdf_file)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to process PDF: {e}"}), 500

        if not raw_text or not raw_text.strip():
            return jsonify({"error": "Extracted syllabus is empty; cannot generate questions."}), 400

        keyword_set = build_keyword_set_from_text(raw_text)

        # Step 3: Text Cleaning + Chunking (500-800 words)
        chunks = chunk_text(raw_text, min_words=500, max_words=800, overlap_words=100)
        
        if not chunks:
            return jsonify({"error": "Could not chunk syllabus text."}), 400

        debug_report = {
            "chunks_created": len(chunks),
            "chunk_word_counts": [_approx_word_count(chunk) for chunk in chunks][:10],
            "raw_text_chars": len(raw_text),
            "raw_text_words": _approx_word_count(raw_text),
            "bloom_api_calls_count": 0,
            "bloom_batch_size": 15,
        }

        # Step 4: GPT Question Generation (NO Bloom here)
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
            return jsonify({"error": "No questions could be generated from the syllabus."}), 422

        debug_report["raw_questions_generated"] = len(all_raw_questions)
        debug_report["raw_questions_per_chunk"] = raw_questions_per_chunk

        # Convert GeneratedQuestion to Question objects for validation
        question_objects = [Question(q.text, q.source_chunk_id) for q in all_raw_questions]

        # Step 5: Question Validation (hard rejection rules)
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
            return jsonify({"error": "No valid questions passed validation."}), 422

        # Step 6: Question Storage (JSON format)
        # Store questions with their text and source info
        stored_questions = []
        for q in valid_questions:
            stored_questions.append({
                "text": q.text,
                "source_chunk_id": q.source_chunk_id,
            })
        
        # Step 7: GPT Bloom Classification (separate call)
        # Classify each question and add Bloom level
        classified_questions = []
        batch_size = 15
        bloom_api_calls = 0
        for start in range(0, len(stored_questions), batch_size):
            batch = stored_questions[start:start + batch_size]
            batch_texts = [q["text"] for q in batch]
            batch_results = classify_bloom_levels_gpt_batch(batch_texts)
            bloom_api_calls += 1

            for stored_q, classification in zip(batch, batch_results):
                if classification is None:
                    classification = classify_bloom_level_heuristic(stored_q["text"])
                    if classification is None:
                        classification = classify_bloom_level_gpt(stored_q["text"])
                        bloom_api_calls += 1
                if classification:
                    q_obj = Question(
                        text=stored_q["text"],
                        source_chunk_id=stored_q["source_chunk_id"],
                    )
                    q_obj.bloom_level = classification.level
                    q_obj.bloom_verb = classification.verb
                    q_obj.marks = _assign_marks_by_bloom(classification.level)
                    classified_questions.append(q_obj)
                # If classification fails, skip the question (strict quality control)

        debug_report["bloom_classified"] = len(classified_questions)
        debug_report["bloom_failed"] = len(stored_questions) - len(classified_questions)
        debug_report["bloom_api_calls_count"] = bloom_api_calls
        debug_report["bloom_batch_size"] = batch_size
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

        if not classified_questions:
            return jsonify({"error": "No questions could be classified with Bloom levels."}), 422

        # Step 8: Constraint-Based Paper Generation
        # Default constraints (can be made configurable via UI later)
        total_marks = 50
        bloom_distribution = {
            "Remember": 0.2,
            "Understand": 0.3,
            "Apply": 0.3,
            "Analyze": 0.15,
            "Evaluate": 0.05,
            "Create": 0.0,  # Usually not in basic exams
        }

        try:
            paper = generate_question_paper(
                pool=classified_questions,
                total_marks=total_marks,
                bloom_distribution=bloom_distribution,
            )
        except ValueError as e:
            return jsonify({"error": f"Could not generate paper: {e}"}), 422
        
        debug_report["paper_total_marks_target"] = total_marks
        debug_report["paper_questions_selected"] = len(paper.get("questions", [])) if paper else 0

        # Step 9: Review & Export (return JSON)
        # Store the generated paper
        questions_storage.append(paper)
        
        if debug_mode:
            return jsonify({"paper": paper, "debug": debug_report}), 200
        return jsonify(paper), 200

    @app.route("/api/questions", methods=["GET"])
    def api_get_questions():
        """Get stored questions (for review)."""
        return jsonify({"questions": questions_storage}), 200

    return app


def _assign_marks_by_bloom(bloom_level: str) -> int:
    """
    Assign marks based on Bloom's Taxonomy level.
    Simple heuristic - can be refined based on faculty feedback.
    """
    marks_map = {
        "Remember": 2,
        "Understand": 2,
        "Apply": 5,
        "Analyze": 5,
        "Evaluate": 10,
        "Create": 10,
    }
    return marks_map.get(bloom_level, 5)  # Default to 5 if unknown


def _approx_word_count(text: str) -> int:
    """Approximate word count using whitespace splitting."""
    if not text:
        return 0
    return len(text.split())


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
