### \# Agentic AI Knowledge \& Research Copilot



A multi-agent AI system for answering cybersecurity questions and generating

structured research reports using the MITRE ATT\&CK knowledge base.



\## What it does



\- \*\*Layer 1 (Quick Answer):\*\* Takes a single question, searches the MITRE ATT\&CK

&#x20; database, and returns a cited answer with a confidence score. Retries once if

&#x20; answer quality is low.

\- \*\*Layer 2 (Research Report):\*\* Takes a complex question, breaks it into

&#x20; sub-questions, runs Layer 1 on each, and synthesizes a full structured report

&#x20; with Executive Summary, Findings, Recommendations, and Confidence score.



\## Tech Stack



| Component | Tool |

|---|---|

| LLM | Groq (llama-3.3-70b-versatile) — free tier |

| Embeddings | sentence-transformers all-MiniLM-L6-v2 — local, no API key |

| Vector DB | Qdrant — runs locally on disk, no Docker needed |

| Orchestration | LangGraph |

| Knowledge Base | MITRE ATT\&CK Enterprise (835 techniques, 1373 chunks) |



\## Project Structure

```

agentic-attack-copilot/

├── README.md

├── requirements.txt

├── .env.example

├── main.py                          # optional entry point

│

├── configs/

│   ├── settings.yaml                # chunk\_size, top\_k, model names, thresholds

│   └── prompts.yaml                 # all prompts for L1 + L2

│

├── data/

│   ├── raw/                         # MITRE ATT\&CK STIX bundle

│   ├── processed/                   # docs.jsonl + chunks.jsonl

│   ├── qdrant\_storage/              # local vector DB (pre-populated, 1373 chunks)

│   └── eval/                        # evaluation questions + expected references

│

├── ingestion/

│   ├── extract\_docs.py              # extracts techniques from STIX bundle

│   └── chunk\_and\_index.py           # chunks text + indexes into Qdrant

│

├── retrieval/

│   ├── qdrant\_client.py             # Qdrant connection + collection setup

│   └── qdrant\_retriever.py          # semantic search → EvidenceChunk objects

│

├── llm/

│   └── providers.py                 # chat() via Groq + embed\_texts() local

│

├── graph/

│   ├── layer1\_graph.py              # LangGraph L1: retrieve→evaluate→generate→reflect→retry

│   └── layer2\_graph.py              # LangGraph L2: decompose→run L1 per sub-Q→synthesize

│

├── layer2\_research/

│   ├── schemas.py                   # Layer2Report dataclass

│   ├── decomposer.py                # breaks question into ≤5 sub-questions

│   ├── researcher.py                # runs Layer 1 on each sub-question

│   ├── synthesizer.py               # writes structured 5-section report

│   └── \_\_init\_\_.py

│

├── app/

│   ├── config.py                    # loads .env + settings.yaml

│   ├── api.py                       # FastAPI: /ask (L1), /research (L2), /health

│   └── ui.py                        # Streamlit: Q\&A tab + Research tab + Evidence viewer

│

├── evaluation/

│   ├── eval\_questions.json          # 10-30 test questions

│   ├── baseline\_rag.py              # plain RAG baseline (no agents)

│   ├── run\_layer1.py                # runs L1 on eval set

│   ├── run\_layer2.py                # runs L2 on eval set

│   ├── metrics.py                   # citation coverage, relevance, groundedness

│   └── compare.py                   # baseline vs agentic comparison report

│

├── scripts/

│   ├── check\_qdrant.py              # verify DB is populated

│   └── search.py                    # test retrieval from command line

│

└── tests/

&#x20;   ├── test\_stix\_loads.py           # ingestion sanity check

&#x20;   ├── test\_retrieval\_shape.py      # retrieval shape check

&#x20;   ├── tests\_l1/

&#x20;   │   └── test\_layer1.py           # Layer 1 unit tests (4 tests)

&#x20;   └── tests\_l2/

&#x20;       └── test\_layer2.py           # Layer 2 unit tests (5 tests)

```



\## Setup



\### Requirements

\- Python 3.10+

\- Free Groq API key from \[console.groq.com](https://console.groq.com)

\- No Docker, no OpenAI key needed



\### Installation



\*\*1. Clone the repo:\*\*

```bash

git clone https://github.com/NityaRavi12/Agentic-AI-Knowledge-Research-Copilot.git

cd Agentic-AI-Knowledge-Research-Copilot

```



\*\*2. Create virtual environment:\*\*

```bash

python -m venv venv

venv\\Scripts\\activate        # Windows

source venv/bin/activate     # Mac/Linux

```



\*\*3. Install dependencies:\*\*

```bash

pip install -r requirements.txt

pip install pytest

```



\*\*4. Create your .env file:\*\*

```bash

cp .env.example .env

```

Add to `.env`:

```

GROQ\_API\_KEY=your\_key\_here

GROQ\_MODEL=llama-3.3-70b-versatile

QDRANT\_URL=http://localhost:6333

```



\*\*5. Verify the database:\*\*

```bash

python scripts/check\_qdrant.py

\# Expected: Points count: 1373

```



\*\*6. Run all tests:\*\*

```bash

python -m pytest tests/ tests\_l1/ tests\_l2/ -v

\# Expected: 9 passed, 2 skipped

```



\## Usage



\### Layer 1 — Quick Answer

```python

from dotenv import load\_dotenv

load\_dotenv()



from graph.layer1\_graph import run\_layer1



result = run\_layer1("How do attackers use PowerShell?")

print(result.answer)

print("Citations:", result.citations)

print("Confidence:", result.confidence)

```



\### Layer 2 — Research Report

```python

from dotenv import load\_dotenv

load\_dotenv()



from layer2\_research.researcher import run\_layer2



report = run\_layer2("How do attackers gain persistence on Windows?")

print(report.report)

print("Confidence:", report.confidence)

```



\## AI Concepts Used



| Concept | Where |

|---|---|

| Embeddings | llm/providers.py → embed\_texts() |

| Vector / Semantic Search | retrieval/qdrant\_retriever.py → search() |

| RAG (Retrieval Augmented Generation) | graph/layer1\_graph.py → node\_generate() |

| Agentic AI with reflection + retry | graph/layer1\_graph.py → node\_reflect() |

| Query Decomposition | layer2\_research/decomposer.py |

| Multi-step synthesis | layer2\_research/synthesizer.py |



```





