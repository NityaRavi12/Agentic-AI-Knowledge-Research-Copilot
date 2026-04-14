# 🛡️ Agentic AI Knowledge & Research Copilot

A multi-agent AI system for answering cybersecurity questions and generating structured research reports using the MITRE ATT&CK knowledge base. Designed to reduce hallucination — answers are grounded in retrieved ATT&CK evidence, filtered for relevance, and checked for grounding before being returned.

---

## What it does

| Layer | Mode | Description |
|---|---|---|
| **Layer 1** | Fact-Checker | Takes a single question → retrieves evidence → filters with a CRAG-inspired LLM relevance check → generates cited answer → evaluates grounding → revises once if needed |
| **Layer 2** | Research Copilot | Takes a complex question → decomposes into sub-questions → runs Layer 1 on each → synthesizes a full structured report with Executive Summary, Findings, Recommendations |

---

## Demo

### Layer 1 — Fact-Checker
> **Q:** How do adversaries use PowerShell for execution?
>
> **A:** Adversaries abuse PowerShell commands and scripts for execution [T1059.001]. They can execute commands directly or indirectly without invoking `powershell.exe` through interfaces to PowerShell's underlying `System.Management.Automation` assembly DLL [T1059.001]. Additionally, adversaries can use PowerShell profiles [T1546.013] to gain persistence...
>
> **Citations:** T1059.001 · T1546.013 · T1086 · **Confidence: 83%**

### Layer 2 — Research Report
> **Q:** How would an attacker move laterally through a corporate network without being detected?
>
> Generates a full structured report with Executive Summary, Findings (with citations), Recommendations, Assumptions & Gaps, and Confidence score.

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) — free tier |
| Embeddings | `sentence-transformers all-MiniLM-L6-v2` — runs locally, no API key |
| Vector DB | Qdrant — local persistent storage, no Docker needed |
| Orchestration | LangGraph |
| UI | Streamlit |
| Knowledge Base | MITRE ATT&CK Enterprise (835 techniques, 1,373 chunks) |

---

## Architecture

### Layer 1 — Agentic RAG Pipeline

```
User Question
    ↓
[Retrieve]               vector search → top-12 chunks from Qdrant
    ↓
[Evaluate Retrieval]     CRAG-inspired LLM filter — scores relevance and sufficiency;
                         removes low-quality chunks; optionally rewrites query for retry
    ↓ (if weak → Prepare Retry → back to Retrieve, max 1 retry)
[Generate]               generates cited answer from approved chunks only
    ↓
[Evaluate Answer]        checks grounding + completeness via LLM self-evaluation
    ↓ (if weak → Revise Answer → re-evaluate, max 1 revision)
[Finalize]
    ↓
Grounded Answer with Citations + Confidence Score
```

### Layer 2 — Autonomous Research Pipeline

```
Complex Research Question
    ↓
[Decompose]              LLM splits into ≤5 focused sub-questions
    ↓
[Run Layer 1 × N]        full Layer 1 pipeline on each sub-question
    ↓
[Synthesize]             LLM writes structured report from all answers
    ↓
Research Report (Executive Summary · Findings · Recommendations · Gaps · Confidence)
```

---

## AI Concepts Used

| Concept | Implementation |
|---|---|
| RAG (Retrieval Augmented Generation) | `graph/layer1_graph.py` → `node_generate()` |
| CRAG-inspired filtering | `graph/layer1_graph.py` → `node_evaluate_retrieval()` — LLM scores relevance and sufficiency; no web search fallback (unlike full CRAG) |
| Self-RAG (Answer Grounding) | `graph/layer1_graph.py` → `node_evaluate_answer()` |
| Answer revision loop | `graph/layer1_graph.py` → `node_revise_answer()` |
| Confidence scoring | `graph/layer1_graph.py` → weighted average of LLM-self-reported scores (relevance, sufficiency, groundedness, completeness) |
| Agentic retry loop | LangGraph conditional edges with max 1 retrieval retry and max 1 answer revision |
| Query decomposition | `layer2_research/decomposer.py` |
| Multi-step synthesis | `layer2_research/synthesizer.py` |
| Local embeddings | `llm/providers.py` → `embed_texts()` |
| Semantic search | `retrieval/qdrant_retriever.py` → `search()` |

---

## Project Structure

```
├── app/
│   ├── ui.py                # Streamlit: Q&A tab + Research tab
│   ├── graph.py             # Simplified Layer 1 graph (legacy)
│   └── config.py            # Settings loader
├── configs/
│   └── settings.yaml        # chunk_size, top_k, thresholds, model names
├── evaluation/
│   ├── eval_questions.json  # 9 test questions with ground truth
│   ├── baseline_rag.py      # Naive RAG baseline for comparison
│   ├── run_layer1.py        # Run Layer 1 eval + baseline
│   ├── run_layer2.py        # Run Layer 2 eval
│   ├── metrics.py           # Embedding-based similarity metrics (local, no API)
│   └── compare.py           # Print comparison table
├── graph/
│   ├── layer1_graph.py      # LangGraph: Retrieve→EvalRetrieval→Generate→EvalAnswer→Revise→Finalize
│   └── layer2_graph.py      # LangGraph: Decompose→L1 calls→Synthesize
├── ingestion/
│   ├── extract_docs.py      # parse MITRE ATT&CK STIX bundle → docs.jsonl
│   └── chunk_and_index.py   # chunk + embed + upsert into Qdrant
├── layer1_qa/
│   ├── planner.py           # converts question → ATT&CK search query
│   ├── answerer.py          # generates cited answer from evidence
│   ├── evaluators.py        # CRAG-inspired relevance + grounding checks
│   ├── reflector.py         # grounding verification
│   ├── pipeline.py          # Layer 1 pipeline orchestration
│   └── schemas.py           # Pydantic schemas for Layer 1
├── layer2_research/
│   ├── decomposer.py        # break question into sub-questions
│   ├── researcher.py        # run Layer 1 on each sub-question
│   ├── synthesizer.py       # synthesize structured report
│   └── schemas.py           # Layer2Report dataclass
├── llm/
│   └── providers.py         # Groq chat + local MiniLM embeddings
├── retrieval/
│   ├── qdrant_client.py     # local persistent Qdrant client
│   └── qdrant_retriever.py  # vector search → EvidenceChunk list
├── scripts/
│   ├── check_qdrant.py      # sanity check collection
│   └── search.py            # CLI retrieval test
├── tests/                   # ingestion + retrieval tests
├── tests_l1/                # Layer 1 unit tests (all mocked)
└── tests_l2/                # Layer 2 unit tests (all mocked)
```

---

## Setup

### Requirements
- Python 3.10+
- Free Groq API key from [console.groq.com](https://console.groq.com) (no credit card)
- No Docker, no OpenAI key needed

### Installation

**1. Clone the repo:**
```bash
git clone https://github.com/NityaRavi12/Agentic-AI-Knowledge-Research-Copilot.git
cd Agentic-AI-Knowledge-Research-Copilot
```

**2. Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download MITRE ATT&CK data:**
```bash
# Mac/Linux
curl -L "https://github.com/mitre/cti/raw/master/enterprise-attack/enterprise-attack.json" \
     -o data/raw/enterprise-attack.json

# Windows
curl -L "https://github.com/mitre/cti/raw/master/enterprise-attack/enterprise-attack.json" -o data/raw/enterprise-attack.json
```

**5. Set up your .env file:**

Copy `.env.example` to `.env` and add your Groq key:
```
GROQ_API_KEY=gsk_...your_key_here...
```

**6. Run ingestion (builds the vector index):**

Mac/Linux:
```bash
PYTHONPATH=. python3 ingestion/extract_docs.py
PYTHONPATH=. python3 ingestion/chunk_and_index.py
```

Windows:
```cmd
set PYTHONPATH=.
python ingestion/extract_docs.py
python ingestion/chunk_and_index.py
```

Expected output: `Wrote 835 docs ... Wrote 1373 chunks ... Indexed chunks into Qdrant.`

**7. Verify retrieval works:**

Mac/Linux:
```bash
PYTHONPATH=. python3 scripts/search.py "PowerShell execution"
```
Windows:
```cmd
python scripts/search.py "PowerShell execution"
```

---

## Running the App

### Streamlit UI

Mac/Linux:
```bash
streamlit run app/ui.py
```
Windows:
```cmd
python -m streamlit run app/ui.py
```
Opens at `http://localhost:8501`

> **Note:** The UI calls the pipeline modules directly in-process. A REST API layer is identified as future work.

---

## Running Tests

Mac/Linux:
```bash
PYTHONPATH=. python3 -m pytest tests/ tests_l1/ tests_l2/ -v
```
Windows:
```cmd
python -m pytest tests/ tests_l1/ tests_l2/ -v
```

All tests are fully mocked — no Groq API key or Qdrant index needed to run them.

---

## Running Evaluation

Install additional evaluation dependencies first:
```bash
pip install sentence-transformers numpy
```

Then run in order:

Mac/Linux:
```bash
PYTHONPATH=. python3 evaluation/run_layer1.py
PYTHONPATH=. python3 evaluation/run_layer2.py
PYTHONPATH=. python3 evaluation/metrics.py
PYTHONPATH=. python3 evaluation/compare.py
```

Windows:
```cmd
set GROQ_API_KEY=gsk_...your_key...
python evaluation/run_layer1.py
python evaluation/run_layer2.py
python evaluation/metrics.py
python evaluation/compare.py
```

This runs 9 test questions through both the agentic pipeline and a naive baseline RAG, then computes similarity metrics (faithfulness, answer relevancy, context precision, context recall) using local embeddings — no API calls required for evaluation.

---

## Academic References

| Paper | Application |
|---|---|
| Self-RAG (Asai et al., ICLR 2024) | Answer evaluation + grounding check |
| CRAG (Gangavarapu et al., IEEE 2025) | Inspired the retrieval evaluator's relevance filtering; web-search fallback from the paper is not implemented |
| Multi-Agent Orchestration (Hariharan et al., IEEE 2025) | LangGraph agent design |
| SCMRAG (ACM AAMAS 2025) | Multi-hop reasoning architecture |
