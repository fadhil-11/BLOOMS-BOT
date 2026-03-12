# BLOOMS BOT – Bloom's Taxonomy Aligned Question Paper Generator

## Overview

BLOOMS BOT is a web-based system that generates **university-style Computer Science exam questions** from a PDF syllabus or lecture notes. It follows a strict pipeline that ensures question quality and proper Bloom's Taxonomy classification.

The system is designed with a **fail-closed philosophy**: bad questions are discarded, not fixed. Each module has a single responsibility, and GPT calls are separated by purpose.

## Tech Stack

- **Backend**: Python, Flask  
- **AI**: OpenAI GPT API (separate calls for generation and classification)  
- **PDF Processing**: `pypdf`  
- **Storage**: In-memory / JSON  
- **Frontend**: HTML + CSS + minimal JavaScript  
- **Config / Secrets**: `.env` file (never committed to Git)

## Pipeline

The system follows this strict pipeline:

1. **PDF Upload** → User uploads syllabus/lecture notes as PDF
2. **PDF Text Extraction** → Extract and clean text from PDF
3. **Text Chunking** → Split into 500-800 word chunks with overlap
4. **GPT Question Generation** → Generate questions from syllabus content (NO Bloom levels)
5. **Question Validation** → Hard rejection of bad questions (forbidden words, quality checks)
6. **Question Storage** → Store validated questions
7. **GPT Bloom Classification** → Separate GPT call to classify Bloom levels
8. **Paper Generation** → Constraint-based selection with Bloom distribution
9. **Export** → Return JSON question paper

## Project Structure

```
bloomsbot/
├── app.py                    # Flask app and HTTP routes
├── pdf_processor.py          # PDF → clean text extraction
├── text_chunker.py           # Text chunking (500-800 words)
├── gpt_question_gen.py       # GPT question generation (no Bloom)
├── question_validator.py     # Strict validation and rejection
├── blooms_classifier.py       # Bloom classification (separate GPT call)
├── paper_generator.py        # Constraint-based paper assembly
├── templates/
│   └── index.html            # Upload UI
├── static/
│   └── styles.css            # Styling
├── .gitignore                # Git ignore rules
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup & Running

### 1. Clone the repository

```bash
git clone <repository-url>
cd bloomsbot
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env  # Windows: copy .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_api_key_here
```

**📖 Need help setting up the API?** See `API_SETUP_GUIDE.md` for detailed instructions.

### 5. Run the application

```bash
python app.py
```

Or using Flask:

```bash
export FLASK_APP=app.py        # Windows (PowerShell): $env:FLASK_APP="app.py"
flask run
```

Visit `http://127.0.0.1:5000` in your browser.

## Design Principles

- **GPT creates meaning** → Questions generated only from syllabus content
- **Code enforces rules** → Strict validation rejects bad questions
- **Bad questions are discarded** → No fixing, only rejection
- **One responsibility per module** → Each file has a single purpose
- **One purpose per GPT call** → Generation and classification are separate

## Module Responsibilities

- **`pdf_processor.py`**: Extracts and cleans text from PDF files
- **`text_chunker.py`**: Splits text into 500-800 word chunks
- **`gpt_question_gen.py`**: Generates questions using GPT (no Bloom classification)
- **`question_validator.py`**: Validates questions and rejects those with forbidden words or poor quality
- **`blooms_classifier.py`**: Classifies questions by Bloom's Taxonomy level (separate GPT call)
- **`paper_generator.py`**: Selects questions to match mark distribution and Bloom constraints

## Validation Rules

Questions are rejected if they contain:
- Forbidden words: "zero", "unlike", "therefore", "pham", "something", "any question"
- Less than 6 meaningful words
- No technical CS nouns
- Missing question mark
- Too short (< 20 characters)

## Security

- **Never commit `.env`** → It's in `.gitignore`
- **API keys are secret** → Store only in `.env` file
- **No hardcoded keys** → All keys read from environment

## Testing

Test your API setup:

```bash
python test_api.py
```

This verifies your OpenAI API key is configured correctly.

## Academic Notes

This system is designed for academic evaluation with:
- **Modular architecture** → Easy to explain and extend
- **Pedagogically grounded** → Follows Bloom's Taxonomy principles
- **Human-in-the-loop** → Questions can be reviewed before use
- **API-driven** → Not dependent on a single AI model

## License

Academic project - use for educational purposes.
