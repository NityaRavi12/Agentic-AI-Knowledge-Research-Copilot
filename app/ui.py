"""
Streamlit UI for the Agentic AI Knowledge & Research Copilot.
Run: streamlit run app/ui.py
"""

import sys
import time
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import streamlit as st

st.set_page_config(
    page_title="ATT&CK Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem !important; }
.title-block {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}
.title-block h1 { color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0 0 0.3rem 0; }
.title-block p { color: #a8c8e8; font-size: 0.95rem; margin: 0; }
.answer-box {
    background: #f8f9fa;
    color: #111827;
    border-left: 4px solid #2c5364;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    white-space: pre-wrap;
    line-height: 1.6;
}
.citation-pill {
    display: inline-block;
    background: #e8f4fd;
    color: #1a5276;
    border: 1px solid #aed6f1;
    border-radius: 20px;
    padding: 2px 10px;
    margin: 3px;
    font-size: 0.85rem;
    font-weight: 600;
}
.stButton > button {
    background: #2c5364 !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🛡️ ATT&CK Copilot")
    st.markdown("---")
    st.markdown("**Pipeline Agents**")
    st.markdown("- 🔍 **Retriever** — semantic evidence search\n"
    "- 🧪 **Retrieval Evaluator** — relevance + sufficiency check\n"
    "- ✍️ **Answerer** — grounded cited answer\n"
    "- 🔎 **Answer Evaluator** — grounding + completeness check\n"
    "- 🔄 **Revision Loop** — retry/revise when evidence is weak")
    st.markdown("---")
    st.markdown("**Data Source**")
    st.markdown("[MITRE ATT&CK Enterprise](https://attack.mitre.org/)")
    st.markdown("835 techniques · 1,373 chunks")
    st.markdown("---")
    st.markdown("**Models**")
    st.markdown("🧠 Chat: `llama-3.3-70b` (Groq)\n📐 Embed: `all-MiniLM-L6-v2` (local)")

st.markdown("""
<div class="title-block">
    <h1>🛡️ Agentic ATT&CK Copilot</h1>
    <p>Hallucination-proof cybersecurity Q&A · Powered by MITRE ATT&CK + Agentic RAG</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Fact-Checker (Layer 1)", "🧬 Research Mode (Layer 2)"])

with tab1:
    st.markdown("#### Ask a factual question about MITRE ATT&CK")
    col_input, col_ex = st.columns([2, 1])
    with col_ex:
        st.markdown("**Try an example:**")
        for ex in ["How do adversaries use PowerShell for execution?","What is credential dumping?","How do attackers maintain persistence on Windows?","What is T1059?","What is T9999?"]:
            if st.button(ex[:55] + ("..." if len(ex) > 55 else ""), key=f"e1_{ex[:10]}"):
                st.session_state["q1"]=ex; st.rerun()
    with col_input:
        q1 = st.text_area("Your question", value=st.session_state.get("q1",""), placeholder="e.g. How do adversaries use PowerShell?", height=120, key="q1i")
        ask_btn = st.button("🔍 Ask", key="ask")
    if ask_btn and q1 and q1.strip():
        with st.spinner("Running agentic pipeline..."):
            start = time.time()
            try:
                from graph.layer1_graph import run_layer1
                result = run_layer1(q1.strip())
                elapsed = round(time.time()-start,2)
                st.markdown("---")
                if result.answer and not result.answer.upper().startswith("INSUFFICIENT"):
                    st.success("✅ Grounded Answer")
                else:
                    st.warning("⚠️ Insufficient Evidence")
                c1,c2,c3 = st.columns(3)
                c1.metric("⏱️ Time",f"{elapsed}s"); c2.metric("🔄 Retries",result.retries); c3.metric("🎯 Confidence",f"{result.confidence:.0%}")
                st.markdown("**Answer**")
                st.markdown(f'<div class="answer-box">{result.answer}</div>', unsafe_allow_html=True)
                if result.citations:
                    st.markdown("**📎 Citations**")
                    st.markdown("".join(f'<span class="citation-pill">{c}</span>' for c in result.citations), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
    elif ask_btn:
        st.warning("Please enter a question.")

with tab2:
    st.markdown("#### Ask a complex analytical question")
    col_input2, col_ex2 = st.columns([2, 1])
    with col_ex2:
        st.markdown("**Try an example:**")
        for ex in ["How would an attacker move laterally through a corporate network without being detected?","Compare initial access techniques and explain how they chain into persistence","What techniques do APT groups use for credential access?"]:
            if st.button(ex[:55] + ("..." if len(ex) > 55 else ""), key=f"e2_{ex[:10]}"):
                st.session_state["q2"]=ex; st.rerun()
    with col_input2:
        q2 = st.text_area("Your research question", value=st.session_state.get("q2",""), placeholder="e.g. How would an attacker move laterally?", height=120, key="q2i")
        max_sub = st.slider("Max sub-questions", 2, 7, 4)
        research_btn = st.button("🧬 Research", key="research")
    if research_btn and q2 and q2.strip():
        with st.spinner(f"Running Layer 2 research pipeline..."):
            start = time.time()
            try:
                from layer2_research.researcher import run_layer2
                report = run_layer2(q2.strip(), max_subquestions=max_sub)
                elapsed = round(time.time()-start,2)
                st.markdown("---")
                c1,c2,c3 = st.columns(3)
                c1.metric("⏱️ Time",f"{elapsed}s"); c2.metric("📋 Sub-questions",len(report.subquestions)); c3.metric("🎯 Confidence",f"{report.confidence:.0%}")
                with st.expander("📋 Sub-questions", expanded=False):
                    for i,sq in enumerate(report.subquestions,1): st.markdown(f"**{i}.** {sq}")
                st.markdown("---")
                st.markdown("📄 **Research Report**")
                report_text = report.report
                if "Layer1Result(" in report_text:
                    cleaned = re.sub(r"Layer1Result\(question='[^']*',\s*answer='((?:[^'\\]|\\.)*)'.*?(?=Layer1Result|\Z)",r"\1\n\n",report_text,flags=re.DOTALL)
                    report_text = cleaned if cleaned.strip() else report_text
                st.markdown(report_text)
            except Exception as e:
                st.error(f"Error: {e}")
    elif research_btn:
        st.warning("Please enter a research question.")