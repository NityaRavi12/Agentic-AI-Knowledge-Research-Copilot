# 🛡️ Agentic AI Knowledge & Research Copilot

A multi-agent AI system for answering cybersecurity questions and generating structured research reports using the MITRE ATT&CK knowledge base. Built to be **hallucination-proof** — every answer is grounded in retrieved evidence and verified before being returned.

---

## What it does

| Layer | Mode | Description |
|---|---|---|
| **Layer 1** | Fact-Checker | Takes a single question → retrieves evidence from ATT&CK → filters with CRAG → generates cited answer → verifies grounding with Self-RAG → retries once if needed |
| **Layer 2** | Research Copilot | Takes a complex question → decomposes into sub-questions → runs Layer 1 on each → synthesizes a full structured report with Executive Summary, Findings, Recommendations |

---

## Demo

### Layer 1 — Fact-Checker
> **Q:** How do adversaries use PowerShell for execution?
>
> **A:** Adversaries abuse PowerShell commands and scripts for execution [T1059.001]. They can execute commands directly or indirectly without invoking `powershell.exe` through interfaces to PowerShell's underlying `System.Management.Automation` assembly DLL [T1059.001]. Additionally, adversaries can use PowerShell profiles [T1504][T1546.013] to gain persistence...
>
> **Citations:** T1059.001 · T1504 · T1546.013 · T1086 · **Confidence: 100%**

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
| API | FastAPI |
| UI | Streamlit |
| Knowledge Base | MITRE ATT&CK Enterprise (835 techniques, 1,373 chunks) |

---

## Architecture

```
User Question
    ↓
[Planner]          distills question into ATT&CK search query
    ↓
[Retriever]        vector search → top-12 chunks from Qdrant
    ↓
[Evaluator]        CRAG-style filter — removes irrelevant chunks
    ↓
[Answerer]         generates cited answer from approved evidence only
    ↓
[Reflector]        Self-RAG grounding check — retries once if ungrounded
    ↓
Grounded Answer with Citations
```

For **Layer 2**, the Decomposer breaks the question into sub-questions, runs the full Layer 1 pipeline on each, and the Synthesizer writes a structured research report.

---

## AI Concepts Used

| Concept | Implementation |
|---|---|
| RAG (Retrieval Augmented Generation) | `graph/layer1_graph.py` → `node_generate()` |
| CRAG (Corrective RAG) | `graph/layer1_graph.py` → `node_evaluate()` |
| Self-RAG (Self-Reflection) | `graph/layer1_graph.py` → `node_reflect()` |
| Agentic retry loop | LangGraph conditional edges with max 1 retry |
| Query decomposition | `layer2_research/decomposer.py` |
| Multi-step synthesis | `layer2_research/synthesizer.py` |
| Local embeddings | `llm/providers.py` → `embed_texts()` |
| Semantic search | `retrieval/qdrant_retriever.py` → `search()` |

---

## Project Structure

```
├── app/
│   ├── api.py               # FastAPI: /ask (L1), /research (L2), /health
│   ├── ui.py                # Streamlit: Q&A tab + Research tab
│   └── config.py            # Settings + prompts loader
├── configs/
│   ├── settings.yaml        # chunk_size, top_k, thresholds, model names
│   └── prompts.yaml         # all prompts for L1 + L2 agents
├── graph/
│   ├── layer1_graph.py      # LangGraph: Planner→Retriever→Evaluator→Answerer→Reflector
│   └── layer2_graph.py      # LangGraph: Decomposer→L1 calls→Synthesizer
├── ingestion/
│   ├── extract_docs.py      # parse MITRE ATT&CK STIX bundle → docs.jsonl
│   └── chunk_and_index.py   # chunk + embed + upsert into Qdrant
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
├── tests_l1/                # Layer 1 contract tests (14 tests)
└── tests_l2/                # Layer 2 tests
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
curl -L "https://github.com/mitre/cti/raw/master/enterprise-attack/enterprise-attack.json" \
     -o data/raw/enterprise-attack.json
```

**5. Set up your .env file:**
```bash
cp .env.example .env
```
Add your Groq key to `.env`:
```
GROQ_API_KEY=gsk_...your_key_here...
```

**6. Run ingestion (builds the vector index):**
```bash
PYTHONPATH=. python3 ingestion/extract_docs.py
PYTHONPATH=. python3 ingestion/chunk_and_index.py
```
Expected output: `Wrote 1373 chunks ... Indexed chunks into Qdrant.`

**7. Verify retrieval works:**
```bash
PYTHONPATH=. python3 scripts/search.py "PowerShell execution"
```

---

## Running the App

### Streamlit UI
```bash
streamlit run app/ui.py
```
Opens at `http://localhost:8501`

### FastAPI Backend
```bash
uvicorn app.api:app --reload
```
- `GET /health`
- `POST /ask` — Layer 1 fact-checker
- `POST /research` — Layer 2 research report

---

## Running Tests
```bash
PYTHONPATH=. python3 -m pytest tests/ tests_l1/ tests_l2/ -v
```

---

## Academic References

| Paper | Application |
|---|---|
| Self-RAG (Asai et al., ICLR 2024) | Reflector agent + retry loop |
| CRAG (Gangavarapu et al., IEEE 2025) | Evaluator agent evidence filtering |
| Multi-Agent Orchestration (Hariharan et al., IEEE 2025) | LangGraph agent design |
| SCMRAG (ACM AAMAS 2025) | Multi-hop reasoning architecture |
