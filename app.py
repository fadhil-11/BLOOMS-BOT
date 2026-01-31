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
from question_validator import validate_question_batch, Question
from blooms_classifier import classify_bloom_level_gpt, BloomClassification
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

        # Step 3: Text Cleaning + Chunking (500-800 words)
        chunks = chunk_text(raw_text, min_words=500, max_words=800, overlap_words=100)
        
        if not chunks:
            return jsonify({"error": "Could not chunk syllabus text."}), 400

        # Step 4: GPT Question Generation (NO Bloom here)
        all_raw_questions = []
        for chunk_id, chunk in enumerate(chunks):
            raw_questions = generate_questions_for_chunk(
                chunk_text=chunk,
                source_chunk_id=chunk_id,
            )
            all_raw_questions.extend(raw_questions)

        if not all_raw_questions:
            return jsonify({"error": "No questions could be generated from the syllabus."}), 422

        # Convert GeneratedQuestion to Question objects for validation
        question_objects = [Question(q.text, q.source_chunk_id) for q in all_raw_questions]

        # Step 5: Question Validation (hard rejection rules)
        valid_questions = validate_question_batch(question_objects)

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
        for stored_q in stored_questions:
            classification = classify_bloom_level_gpt(stored_q["text"])
            if classification:
                # Create a Question object with Bloom classification
                q_obj = Question(
                    text=stored_q["text"],
                    source_chunk_id=stored_q["source_chunk_id"]
                )
                q_obj.bloom_level = classification.level
                q_obj.bloom_verb = classification.verb
                # Assign marks based on Bloom level (simple heuristic)
                q_obj.marks = _assign_marks_by_bloom(classification.level)
                classified_questions.append(q_obj)
            # If classification fails, skip the question (strict quality control)

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

        # Step 9: Review & Export (return JSON)
        # Store the generated paper
        questions_storage.append(paper)
        
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
        "Understand": 3,
        "Apply": 5,
        "Analyze": 7,
        "Evaluate": 10,
        "Create": 12,
    }
    return marks_map.get(bloom_level, 5)  # Default to 5 if unknown


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
