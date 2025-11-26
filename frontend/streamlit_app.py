import streamlit as st
import requests
import subprocess
import os
from typing import List, Dict, Any
from datetime import datetime
import base64

# ===========================
# Configuration
# ===========================
API_BASE_URL = "http://localhost:8000"
DATA_RAW_PATH = "data/raw"
INGEST_SCRIPT = "ingest/ingest_final.py"

st.set_page_config(
    page_title="CliniChat - Medical Assistant",
    page_icon="💊",
    layout="wide",
)

# ===========================
# Session State
# ===========================
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []
if "query_history" not in st.session_state:
    st.session_state.query_history: List[Dict[str, Any]] = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "total_latency" not in st.session_state:
    st.session_state.total_latency = 0.0
if "show_menu" not in st.session_state:
    st.session_state.show_menu = False
if "active_view" not in st.session_state:
    st.session_state.active_view = "chat"  # chat | upload | history | viewer
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

# ===========================
# CSS – giao diện kiểu RoboClinic / CliniChat
# ===========================
st.markdown(
    """
    <style>
    /* Ẩn sidebar mặc định */
    [data-testid="stSidebar"] { display: none !important; }

    .stApp {
        background: #f3f5f9;
        color: #343741;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif;
    }

    /* Top bar */
    .top-bar {
        display: flex;
        align-items: center;
        padding: 18px 40px 8px 40px;
    }
    .hamburger-btn {
        border-radius: 999px;
        border: 1px solid #d7d9e0;
        background: #ffffff;
        width: 46px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        cursor: pointer;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.12);
    }
    .clinichat-logo {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: 0.03em;
        background: linear-gradient(90deg, #7b5cff, #ff7ac3);
        -webkit-background-clip: text;
        color: transparent;
        margin-left: 16px;
    }

    /* Side menu (simple column dưới nút ☰) */
    .menu-panel {
        margin-left: 40px;
        margin-top: 6px;
        width: 220px;
        padding: 12px 12px 14px 12px;
        border-radius: 18px;
        background: #f1f2f6;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
    }

    /* Hero title center */
    .hero-wrapper {
        text-align: center;
        margin-top: 40px;
    }
    .hero-title {
        font-size: 32px;
        font-weight: 600;
        color: #3d4048;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 17px;
        color: #7a7d88;
    }

    /* Suggestion chips */
    .chip-row {
        display: flex;
        justify-content: center;
        gap: 24px;
        margin-top: 32px;
        margin-bottom: 8px;
    }
    .chip-btn {
        border-radius: 999px;
        padding: 10px 22px;
        border: 1px solid #e1e3ec;
        background: #ffffff;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        font-size: 14px;
        color: #4a4d57;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
        white-space: nowrap;
    }

    /* Chat bubbles */
    .chat-container {
        max-width: 900px;
        margin: 24px auto 120px auto;
    }
    .chat-message {
        padding: 10px 14px;
        border-radius: 16px;
        margin-bottom: 10px;
        font-size: 15px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }
    .chat-user {
        background: #ffffff;
        align-self: flex-end;
        border-radius: 16px 16px 4px 16px;
    }
    .chat-assistant {
        background: #eef1f7;
        border-radius: 16px 16px 16px 4px;
    }

    /* Disclaimer box */
    .warning-box {
        background-color: #fff6e6;
        padding: 0.8rem;
        border-radius: 10px;
        border-left: 3px solid #f2a500;
        color: #8a6d3b;
        font-size: 14px;
    }

    /* Buttons chung */
    .stButton>button {
        border-radius: 999px;
        border: 1px solid #dadce5;
        background: #ffffff;
        color: #4a4d57;
        font-size: 14px;
        padding: 8px 14px;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #7b5cff;
        color: #ffffff;
        border-color: #7b5cff;
        box-shadow: 0 6px 18px rgba(123, 92, 255, 0.35);
    }

    /* Chat input dưới cùng */
    [data-testid="stChatInputContainer"] {
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        padding: 10px 200px;
        background: #f3f5f9;
        border-top: 1px solid #dde0ea;
    }
    [data-testid="stChatInputContainer"] textarea {
        border-radius: 999px;
        border: 1px solid #dde0ea !important;
        background: #ffffff !important;
        padding: 10px 18px !important;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.1);
    }

    /* Section tiêu đề nhỏ */
    .section-title {
        font-weight: 600;
        margin-top: 16px;
        margin-bottom: 6px;
        color: #555865;
    }

    .source-card {
        background: #ffffff;
        padding: 6px 10px;
        border-radius: 10px;
        font-size: 13px;
        border: 1px solid #e0e2ea;
        margin-bottom: 6px;
    }

    /* Footer */
    .footer-meta {
        margin-top: 4px;
        text-align: center;
        color: #9a9eaa;
        font-size: 11px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================
# Helper Functions
# ===========================
def check_backend_health() -> bool:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def query_rag(question: str, top_k: int = 3) -> Dict[str, Any] | None:
    try:
        resp = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "top_k": top_k},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Error calling backend: {e}")
        return None


def save_uploaded_file(uploaded_file):
    try:
        os.makedirs(DATA_RAW_PATH, exist_ok=True)
        file_path = os.path.join(DATA_RAW_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path, None
    except Exception as e:
        return None, str(e)


def run_ingestion():
    try:
        result = subprocess.run(
            ["python", INGEST_SCRIPT],
            capture_output=True,
            text=True,
            timeout=300,
        )
        ok = result.returncode == 0
        return ok, result.stdout if ok else result.stderr
    except subprocess.TimeoutExpired:
        return False, "Ingestion timeout (>5 minutes)"
    except Exception as e:
        return False, str(e)


def get_existing_pdfs():
    if not os.path.exists(DATA_RAW_PATH):
        return []
    return sorted(
        [f for f in os.listdir(DATA_RAW_PATH) if f.lower().endswith(".pdf")]
    )


def add_to_history(question: str, answer: str | None):
    st.session_state.query_history.insert(
        0,
        {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        },
    )
    st.session_state.query_history = st.session_state.query_history[:50]


def display_pdf(file_path: str):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" height="600" type="application/pdf"></iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Cannot display PDF: {e}")


# ===========================
# TOP BAR
# ===========================
st.markdown('<div class="top-bar">', unsafe_allow_html=True)
col_top1, col_top2 = st.columns([0.12, 0.88])
with col_top1:
    if st.button("☰", key="hamburger", help="Menu"):
        st.session_state.show_menu = not st.session_state.show_menu

with col_top2:
    st.markdown('<span class="clinichat-logo">CliniChat</span>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# MENU dưới nút ☰
if st.session_state.show_menu:
    st.markdown('<div class="menu-panel">', unsafe_allow_html=True)
    m_new = st.button("✨ New Chat", key="menu_new")
    m_upload = st.button("📤 Upload", key="menu_upload")
    m_viewer = st.button("📄 PDF Viewer", key="menu_viewer")
    m_hist = st.button("🕒 History", key="menu_hist")
    st.markdown("</div>", unsafe_allow_html=True)

    if m_new:
        st.session_state.active_view = "chat"
        st.session_state.messages = []
    if m_upload:
        st.session_state.active_view = "upload"
    if m_viewer:
        st.session_state.active_view = "viewer"
    if m_hist:
        st.session_state.active_view = "history"

# ===========================
# HERO + QUICK QUESTIONS
# ===========================
st.markdown(
    """
    <div class="hero-wrapper">
        <div class="hero-title">Talk to Your AI Health Assistant</div>
        <div class="hero-subtitle">
            Ask anything about symptoms, medications, or healthcare.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 3 câu hỏi gợi ý như screenshot
q1 = "What medicine forms are there?"
q2 = "How should I store my medicines?"
q3 = "What if I forget my medicine?"

st.markdown('<div class="chip-row">', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3, gap="large")
with c1:
    if st.button(f"💬  {q1}", key="chip1"):
        st.session_state.selected_question = q1
        st.session_state.active_view = "chat"
with c2:
    if st.button(f"💬  {q2}", key="chip2"):
        st.session_state.selected_question = q2
        st.session_state.active_view = "chat"
with c3:
    if st.button(f"💬  {q3}", key="chip3"):
        st.session_state.selected_question = q3
        st.session_state.active_view = "chat"
st.markdown("</div>", unsafe_allow_html=True)

# MEDICAL DISCLAIMER
with st.expander("⚠️ Medical disclaimer"):
    st.markdown(
        """
        <div class="warning-box">
        This tool provides general information only. It is <strong>not a substitute for professional medical advice,
        diagnosis, or treatment.</strong> Always consult a licensed healthcare professional for medical decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Backend status nhỏ
is_healthy = check_backend_health()
status = "✅ Online" if is_healthy else "❌ Offline"
st.caption(f"Backend status: {status}")

# ===========================
# MAIN VIEW (Chat / Upload / History / Viewer)
# ===========================
active = st.session_state.active_view

# ---- CHAT VIEW ----
if active == "chat":
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if st.session_state.messages:
        st.markdown('<div class="section-title">Conversation</div>', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-message chat-user"><strong>You:</strong> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            data = msg["content"]
            st.markdown(
                f'<div class="chat-message chat-assistant">{data["answer"]}</div>',
                unsafe_allow_html=True,
            )
            # nguồn
            if data.get("sources"):
                with st.expander(f"📚 {len(data['sources'])} sources"):
                    for src in data["sources"]:
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <strong>#{src['id']}</strong> {src['source']} (p.{src['page']})<br>
                                <small>{src['snippet'][:180]}...</small>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            st.caption(f"⏱ {data.get('latency_ms', 0):.0f} ms")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- UPLOAD VIEW ----
elif active == "upload":
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload PDF Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) selected")
        if st.button("🚀 Process documents"):
            progress = st.progress(0)
            status_text = st.empty()
            saved = 0
            for i, f in enumerate(uploaded_files):
                status_text.text(f"Saving {f.name}...")
                path, err = save_uploaded_file(f)
                if not err:
                    saved += 1
                progress.progress((i + 1) / (len(uploaded_files) + 1))
            if saved > 0:
                status_text.text("Running ingestion...")
                progress.progress(0.7)
                success, out = run_ingestion()
                progress.progress(1.0)
                if success:
                    st.success("✅ Ingestion complete!")
                else:
                    st.error("❌ Ingestion failed")
                    with st.expander("Error log"):
                        st.code(out)
            status_text.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---- HISTORY VIEW ----
elif active == "history":
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Query History</div>', unsafe_allow_html=True)

    if st.session_state.query_history:
        for item in st.session_state.query_history[:30]:
            ts = datetime.fromisoformat(item["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"**{ts}** – {item['question']}")
            if item.get("answer"):
                with st.expander("Show answer"):
                    st.markdown(
                        f'<div class="source-card">{item["answer"]}</div>',
                        unsafe_allow_html=True,
                    )
            st.divider()
    else:
        st.info("No history yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---- PDF VIEWER ----
elif active == "viewer":
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">PDF Viewer</div>', unsafe_allow_html=True)

    pdfs = get_existing_pdfs()
    if pdfs:
        selected_pdf = st.selectbox("Select PDF", pdfs)
        if selected_pdf:
            pdf_path = os.path.join(DATA_RAW_PATH, selected_pdf)
            display_pdf(pdf_path)
    else:
        st.info("No PDFs available. Upload documents first.")
    st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# CHAT INPUT (luôn ở dưới cùng)
# ===========================
question = st.chat_input("Type or share photo")

# Nếu user click chip suggestion
if st.session_state.selected_question:
    question = st.session_state.selected_question
    st.session_state.selected_question = None

if question:
    if not is_healthy:
        st.error("Backend offline. Start FastAPI backend first.")
    else:
        # lưu user question
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            result = query_rag(question, top_k=3)

        if result:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": {
                        "answer": result["answer"],
                        "sources": result.get("sources", []),
                        "latency_ms": result.get("latency_ms", 0),
                    },
                }
            )
            st.session_state.query_count += 1
            st.session_state.total_latency += result.get("latency_ms", 0)
            add_to_history(question, result["answer"])
        st.rerun()

# ===========================
# Footer
# ===========================
st.markdown(
    """
    <div class="footer-meta">
        © 2025 CliniChat – RAG Medical Assistant (Llama 3.2 × ChromaDB)
    </div>
    """,
    unsafe_allow_html=True,
)
