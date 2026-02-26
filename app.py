import os
import streamlit as st
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ITEC 3310 Course Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Page background */
.stApp {
    background: #0f1117;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 760px; }

/* Hero header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid #1e2330;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #e8ecf4;
    margin: 0 0 0.4rem;
    letter-spacing: -0.02em;
}
.hero .subtitle {
    font-size: 0.9rem;
    color: #6b7694;
    font-weight: 300;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.hero .badge {
    display: inline-block;
    background: #1a2540;
    border: 1px solid #2a3a60;
    color: #7a9fd4;
    font-size: 0.75rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    margin-top: 0.75rem;
    font-weight: 500;
}

/* Chat messages */
.stChatMessage {
    background: transparent !important;
    border: none !important;
}

[data-testid="stChatMessageContent"] {
    background: #161b2e !important;
    border: 1px solid #1e2744 !important;
    border-radius: 12px !important;
    color: #cdd5e8 !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    padding: 0.9rem 1.1rem !important;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: #1a2d4a !important;
    border-color: #2a4470 !important;
    color: #dde4f5 !important;
}

/* Source citation box */
.source-box {
    background: #0d1420;
    border: 1px solid #1a2744;
    border-left: 3px solid #3a6bc4;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-top: 0.75rem;
    font-size: 0.78rem;
    color: #5a7ab4;
    font-style: italic;
}

/* Chat input */
.stChatInputContainer {
    background: #161b2e !important;
    border: 1px solid #1e2744 !important;
    border-radius: 12px !important;
}
.stChatInputContainer textarea {
    color: #cdd5e8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}

/* Suggestion chips */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.75rem 0 1.5rem;
    justify-content: center;
}
.chip {
    background: #161b2e;
    border: 1px solid #1e2744;
    color: #7a9fd4;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    font-size: 0.78rem;
    cursor: pointer;
    transition: all 0.15s;
}
.chip:hover { background: #1a2744; border-color: #3a6bc4; color: #a8c4e8; }

/* Status / info bar */
.status-bar {
    text-align: center;
    font-size: 0.75rem;
    color: #3a4a6a;
    margin-bottom: 1rem;
    letter-spacing: 0.03em;
}
.status-dot {
    display: inline-block;
    width: 6px; height: 6px;
    background: #2d8a4e;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}

/* Spinner */
.stSpinner > div { border-top-color: #3a6bc4 !important; }
</style>
""", unsafe_allow_html=True)

# ── RAG setup ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building knowledge base…")
def build_pipeline():
    """
    Load the syllabus, chunk it, embed with a lightweight local model,
    store in an in-memory Chroma vector store, then return a retriever.
    We use sentence-transformers (no API key needed for embeddings) and
    Anthropic Claude for generation.
    """
    # All imports use dedicated sub-packages (langchain 0.2+ / LCEL pattern)
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    # 1. LOAD
    doc_path = Path(__file__).parent / "itec3310_syllabus.txt"
    if not doc_path.exists():
        raise FileNotFoundError(f"Syllabus not found at: {doc_path}. Files in parent: {list(Path(__file__).parent.iterdir())}")
    loader = TextLoader(str(doc_path))
    documents = loader.load()

    # 2. CHUNK
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)

    # 3. EMBED (local, CPU, no API key needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # 4. INDEX
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 5. LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        temperature=0,
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        max_output_tokens=1024,
    )

    # 6. PROMPT (LCEL-style ChatPromptTemplate)
    prompt = ChatPromptTemplate.from_template("""You are a helpful course assistant for ITEC 3310 — Data Management and Analytics.
Answer the student's question using ONLY the course syllabus content provided below.
Be concise, friendly, and precise. If the answer is not in the syllabus, say clearly:
"I can't find that in the syllabus — please check with Dr. Nguyen directly."
Do NOT make up policies or dates.

Syllabus content:
{context}

Student question: {question}

Answer:""")

    # 7. LCEL CHAIN  (replaces deprecated RetrievalQA)
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(chain_and_retriever, question: str) -> tuple[str, list]:
    chain, retriever = chain_and_retriever
    answer = chain.invoke(question).strip()
    sources = retriever.invoke(question)
    return answer, sources


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📚 ITEC 3310 Assistant</h1>
    <div class="subtitle">Data Management & Analytics · Fall 2026</div>
    <div class="badge">Powered by Claude · RAG Pipeline Demo</div>
</div>
""", unsafe_allow_html=True)

# ── Build pipeline ─────────────────────────────────────────────────────────────
try:
    qa_chain = build_pipeline()  # returns (chain, retriever) tuple
    st.markdown("""
    <div class="status-bar">
        <span class="status-dot"></span>
        Knowledge base ready · ITEC 3310 Syllabus indexed
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"⚠️ Could not build pipeline: {e}")
    st.info("Make sure `GOOGLE_API_KEY` is set in your Streamlit secrets.")
    st.stop()

# ── Suggestion chips ───────────────────────────────────────────────────────────
SUGGESTIONS = [
    "What's the late assignment policy?",
    "How is the final grade calculated?",
    "When is the midterm exam?",
    "Can I use AI tools for assignments?",
    "How do I contact the instructor?",
]

if not st.session_state.messages:
    st.markdown('<div class="chip-row">' +
        "".join(f'<span class="chip">{s}</span>' for s in SUGGESTIONS) +
        '</div>', unsafe_allow_html=True)

    cols = st.columns(len(SUGGESTIONS))
    for i, (col, suggestion) in enumerate(zip(cols, SUGGESTIONS)):
        with col:
            if st.button(suggestion, key=f"chip_{i}", use_container_width=True,
                         help=suggestion, type="secondary"):
                st.session_state.pending_question = suggestion
                st.rerun()

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source"):
            st.markdown(f'<div class="source-box">📄 Source: {msg["source"]}</div>',
                        unsafe_allow_html=True)

# ── Handle input ───────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about ITEC 3310…")

# Resolve pending chip click or typed input
question = st.session_state.pending_question or user_input
if st.session_state.pending_question:
    st.session_state.pending_question = None

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching syllabus…"):
            try:
                answer, sources = ask(qa_chain, question)
                source_label = None
                if sources:
                    source_label = "ITEC 3310 Syllabus — " + (
                        sources[0].page_content[:80].replace("\n", " ").strip() + "…"
                    )
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
                source_label = None

        st.markdown(answer)
        if source_label:
            st.markdown(f'<div class="source-box">📄 {source_label}</div>',
                        unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "source": source_label,
    })

# ── Footer ─────────────────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; margin-top:2rem; font-size:0.72rem; color:#2a3a5a;">
        Answers are grounded in the ITEC 3310 syllabus only · Always verify with Dr. Nguyen for official decisions
    </div>
    """, unsafe_allow_html=True)
