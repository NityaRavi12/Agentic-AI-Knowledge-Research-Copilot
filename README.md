# Agentic AI Knowledge & Research Copilot

A multi-agent AI system for answering cybersecurity questions and generating structured research reports using the MITRE ATT&CK knowledge base.

## What it does

- **Layer 1 (Quick Answer):** Takes a single question, searches the MITRE ATT&CK database, and returns a cited answer with a confidence score. Retries once if answer quality is low.
- **Layer 2 (Research Report):** Takes a complex question, breaks it into sub-questions, runs Layer 1 on each, and synthesizes a full structured report with Executive Summary, Findings, Recommendations, and Confidence score.

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq (llama-3.3-70b-versatile) — free tier |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 — local, no API key |
| Vector DB | Qdrant — runs locally on disk, no Docker needed |
| Orchestration | LangGraph |
| Knowledge Base | MITRE ATT&CK Enterprise (835 techniques, 1373 chunks) |

## Project Structure

    agentic-attack-copilot/
    ├── README.md
    ├── requirements.txt
    ├── .env.example
    ├── main.py
    ├── configs/
    │   ├── settings.yaml
    │   └── prompts.yaml
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   ├── qdrant_storage/
    │   └── eval/
    ├── ingestion/
    │   ├── extract_docs.py
    │   └── chunk_and_index.py
    ├── retrieval/
    │   ├── qdrant_client.py
    │   └── qdrant_retriever.py
    ├── llm/
    │   └── providers.py
    ├── graph/
    │   ├── layer1_graph.py
    │   └── layer2_graph.py
    ├── layer2_research/
    │   ├── schemas.py
    │   ├── decomposer.py
    │   ├── researcher.py
    │   ├── synthesizer.py
    │   └── __init__.py
    ├── app/
    │   ├── config.py
    │   ├── api.py
    │   └── ui.py
    ├── evaluation/
    │   ├── eval_questions.json
    │   ├── baseline_rag.py
    │   ├── run_layer1.py
    │   ├── run_layer2.py
    │   ├── metrics.py
    │   └── compare.py
    ├── scripts/
    │   ├── check_qdrant.py
    │   └── search.py
    └── tests/
        ├── test_stix_loads.py
        ├── test_retrieval_shape.py
        ├── tests_l1/
        │   └── test_layer1.py
        └── tests_l2/
            └── test_layer2.py

## Setup

### Requirements
- Python 3.10+
- Free Groq API key from [console.groq.com](https://console.groq.com)
- No Docker, no OpenAI key needed

### Installation

**1. Clone the repo:**

    git clone https://github.com/NityaRavi12/Agentic-AI-Knowledge-Research-Copilot.git
    cd Agentic-AI-Knowledge-Research-Copilot

**2. Create virtual environment:**

    python -m venv venv
    venv\Scripts\activate        # Windows
    source venv/bin/activate     # Mac/Linux

**3. Install dependencies:**

    pip install -r requirements.txt
    pip install pytest

**4. Create your .env file:**

    cp .env.example .env

Add to `.env`:

    GROQ_API_KEY=your_key_here
    GROQ_MODEL=llama-3.3-70b-versatile
    QDRANT_URL=http://localhost:6333

**5. Verify the database:**

    python scripts/check_qdrant.py
    # Expected: Points count: 1373

**6. Run all tests:**

    python -m pytest tests/ tests_l1/ tests_l2/ -v
    # Expected: 9 passed, 2 skipped

## Usage

### Layer 1 — Quick Answer

    from dotenv import load_dotenv
    load_dotenv()

    from graph.layer1_graph import run_layer1

    result = run_layer1("How do attackers use PowerShell?")
    print(result.answer)
    print("Citations:", result.citations)
    print("Confidence:", result.confidence)

### Layer 2 — Research Report

    from dotenv import load_dotenv
    load_dotenv()

    from layer2_research.researcher import run_layer2

    report = run_layer2("How do attackers gain persistence on Windows?")
    print(report.report)
    print("Confidence:", report.confidence)

## AI Concepts Used

| Concept | Where |
|---|---|
| Embeddings | llm/providers.py → embed_texts() |
| Vector / Semantic Search | retrieval/qdrant_retriever.py → search() |
| RAG (Retrieval Augmented Generation) | graph/layer1_graph.py → node_generate() |
| Agentic AI with reflection + retry | graph/layer1_graph.py → node_reflect() |
| Query Decomposition | layer2_research/decomposer.py |
| Multi-step synthesis | layer2_research/synthesizer.py |


## What's Next

- [ ] `app/api.py` — FastAPI backend
- [ ] `app/ui.py` — Streamlit UI
- [ ] `evaluation/` — RAGAS/TruLens evaluation vs baseline RAG
- [ ] `configs/prompts.yaml` — extract prompts from code
- [ ] `data/eval/` — evaluation question set
