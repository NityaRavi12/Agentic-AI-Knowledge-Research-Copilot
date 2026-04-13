"""
Streamlit UI for the Agentic AI Knowledge & Research Copilot.
Run: streamlit run app/ui.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env BEFORE anything else
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ATT&CK Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    code, .stCode { font-family: 'JetBrains Mono', monospace; }

    .main { background-color: #0d1117; }
    .block-container { padding-top: 2rem; }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #58a6ff;
        letter-spacing: -0.5px;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #8b949e;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 3px solid #58a6ff;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.88rem;
    }
    .citation-card a { color: #58a6ff; text-decoration: none; }
    .citation-card a:hover { text-decoration: underline; }
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .status-success { background: #1a4731; color: #3fb950; }
    .status-insufficient { background: #3d1f00; color: #f0883e; }
    .status-ungrounded { background: #3d1f1f; color: #f85149; }
    .status-error { background: #2d0f0f; color: #f85149; }
    .subq-item {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin: 0.25rem 0;
        font-size: 0.88rem;
        color: #c9d1d9;
    }
    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #238636, #3fb950);
        margin-top: 4px;
    }
    .stTextArea textarea {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stButton > button {
        background: #238636 !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 0.4rem 1.2rem !important;
    }
    .stButton > button:hover { background: #2ea043 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ ATT&CK Copilot")
    st.markdown("---")
    st.markdown("**Architecture**")
    st.markdown("""
- 🔍 **Planner** — query distillation
- 🧪 **Evaluator** — CRAG evidence filter
- ✍️ **Answerer** — cited answer generation
- 🔎 **Reflector** — Self-RAG grounding check
- 🔄 **Retry** — auto-retry if ungrounded
    """)
    st.markdown("---")
    st.markdown("**Data source**")
    st.markdown("[MITRE ATT&CK Enterprise](https://attack.mitre.org/)")
    st.markdown("835 techniques · 1,373 chunks")
    st.markdown("---")
    st.markdown("**Models**")
    st.markdown("🧠 Chat: `llama-3.3-70b` (Groq)")
    st.markdown("📐 Embed: `all-MiniLM-L6-v2` (local)")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🛡️ ATT&CK Copilot</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Hallucination-proof cybersecurity Q&A · Powered by MITRE ATT&CK + Agentic RAG</p>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Fact-Checker (Layer 1)", "🧬 Research Mode (Layer 2)"])


# ── Helper functions ──────────────────────────────────────────────────────────

def status_badge(status: str) -> str:
    cls = {
        "SUCCESS": "status-success",
        "INSUFFICIENT_EVIDENCE": "status-insufficient",
        "UNGROUNDED_AFTER_RETRY": "status-ungrounded",
        "ERROR": "status-error",
    }.get(status, "status-error")
    label = {
        "SUCCESS": "✅ Grounded Answer",
        "INSUFFICIENT_EVIDENCE": "⚠️ Insufficient Evidence",
        "UNGROUNDED_AFTER_RETRY": "🔴 Ungrounded After Retry",
        "ERROR": "❌ Error",
    }.get(status, status)
    return f'<span class="status-badge {cls}">{label}</span>'


def render_citations(citations: list[dict]):
    if not citations:
        return
    st.markdown("**📎 Citations**")
    for c in citations:
        st.markdown(
            f'<div class="citation-card">'
            f'<strong>{c["technique_id"]}</strong> — {c["title"]}<br>'
            f'<a href="{c["url"]}" target="_blank">{c["url"]}</a>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Tab 1: Fact-Checker ───────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Ask a question about MITRE ATT&CK")
    st.markdown("The system retrieves evidence, filters it, generates a cited answer, and verifies grounding.")

    example_questions = [
        "How do adversaries use PowerShell for execution?",
        "What is credential dumping and how can it be detected?",
        "How do attackers maintain persistence on Windows systems?",
        "What is T1059 and what are its sub-techniques?",
        "What is T9999?",  # should return insufficient evidence
    ]

    col1, col2 = st.columns([3, 1])
    with col1:
        q1 = st.text_area(
            "Your question",
            placeholder="e.g. How do adversaries use PowerShell for execution?",
            height=100,
            key="q1",
        )
    with col2:
        st.markdown("**Try an example:**")
        for eq in example_questions:
            if st.button(eq[:45] + ("..." if len(eq) > 45 else ""), key=f"ex_{eq[:20]}"):
                st.session_state["q1_prefill"] = eq
                st.rerun()

    # Handle prefill from example buttons
    if "q1_prefill" in st.session_state:
        q1 = st.session_state.pop("q1_prefill")

    ask_btn = st.button("🔍 Ask", key="ask_btn", use_container_width=False)

    if ask_btn and q1 and q1.strip():
        with st.spinner("Running agentic pipeline..."):
            start = time.time()
            try:
                from graph.layer1_graph import run_layer1
                result = run_layer1(q1.strip())
                elapsed = round(time.time() - start, 2)

                st.markdown("---")
                status = "✅ Grounded Answer" if result.answer and not result.answer.startswith("INSUFFICIENT") else "⚠️ Insufficient Evidence"
                st.markdown(f"**{status}**")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("⏱️ Time", f"{elapsed}s")
                col_b.metric("🔄 Retries", result.retries)
                col_c.metric("🎯 Confidence", f"{result.confidence:.0%}")

                st.markdown("**Answer**")
                st.markdown(result.answer)

                if result.citations:
                    st.markdown("**📎 Citations**")
                    for cite in result.citations:
                        st.markdown(f"- {cite}")

            except Exception as e:
                st.error(f"Error: {e}")

    elif ask_btn:
        st.warning("Please enter a question.")


# ── Tab 2: Research Mode ──────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Ask a complex analytical question")
    st.markdown("Layer 2 breaks your question into sub-questions, answers each with Layer 1, then synthesizes a full report.")

    example_research = [
        "How would an attacker move laterally through a corporate network without being detected?",
        "Compare initial access techniques and explain how they chain into persistence",
        "What techniques do APT groups use for credential access and how can defenders detect them?",
    ]

    col3, col4 = st.columns([3, 1])
    with col3:
        q2 = st.text_area(
            "Your research question",
            placeholder="e.g. How would an attacker move laterally through a corporate network?",
            height=100,
            key="q2",
        )
    with col4:
        st.markdown("**Try an example:**")
        for eq in example_research:
            if st.button(eq[:45] + "...", key=f"rex_{eq[:20]}"):
                st.session_state["q2_prefill"] = eq
                st.rerun()

    if "q2_prefill" in st.session_state:
        q2 = st.session_state.pop("q2_prefill")

    max_sub = st.slider("Max sub-questions", min_value=2, max_value=7, value=4)
    research_btn = st.button("🧬 Research", key="research_btn")

    if research_btn and q2 and q2.strip():
        with st.spinner(f"Running Layer 2 research pipeline (up to {max_sub} sub-questions)..."):
            start = time.time()
            try:
                from layer2_research.researcher import run_layer2

                report = run_layer2(q2.strip(), max_subquestions=max_sub)
                elapsed = round(time.time() - start, 2)

                st.markdown("---")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("⏱️ Time", f"{elapsed}s")
                col_b.metric("📋 Sub-questions", len(report.subquestions))
                col_c.metric("🎯 Confidence", f"{report.confidence:.0%}")

                # Sub-questions
                with st.expander("📋 Sub-questions answered", expanded=False):
                    for i, sq in enumerate(report.subquestions, 1):
                        st.markdown(f'<div class="subq-item">{i}. {sq}</div>', unsafe_allow_html=True)

                # Report
                st.markdown("**📄 Research Report**")
                # Clean up raw Layer1Result objects if present
                report_text = report.report
                if "Layer1Result(" in report_text:
                    import re
                    # Extract just answer fields
                    answers = re.findall(r"answer='([^']*(?:''[^']*)*)'", report_text)
                    if answers:
                        report_text = "\n\n".join(answers)
                import re
                report_text = report.report
                # Strip raw Layer1Result objects - extract just the answer text
                if "Layer1Result(" in report_text:
                    # Extract answer values from Layer1Result objects
                    cleaned = re.sub(
                        r"Layer1Result\(question='[^']*',\s*answer='((?:[^'\\]|\\.)*)'.*?(?=Layer1Result|\Z)",
                        r"\1\n\n",
                        report_text,
                        flags=re.DOTALL
                    )
                    report_text = cleaned if cleaned.strip() else report_text
                st.markdown(report_text)

                # Sub-results evidence
                if hasattr(report, "subresults") and report.subresults:
                    with st.expander("🔍 Evidence per sub-question", expanded=False):
                        for i, (sq, sr) in enumerate(zip(report.subquestions, report.subresults), 1):
                            st.markdown(f"**{i}. {sq}**")
                            if hasattr(sr, "citations"):
                                st.markdown(str(sr))
                            elif hasattr(sr, "answer"):
                                st.markdown(f"_{sr.answer[:300]}..._")
                            st.markdown("---")

            except Exception as e:
                st.error(f"Error: {e}")

    elif research_btn:
        st.warning("Please enter a research question.")
