import streamlit as st  # Import Streamlit for building the web interface
import sys  # Import sys to manipulate the Python runtime environment
import os  # Import os for file path and environment variable management
import json  # Import json for handling structured data exchange
import time  # Import time for tracking performance metrics

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
# This must be the very first Streamlit call to configure browser tab metadata
st.set_page_config(
    page_title="Medical RAG – Q&A System",  # Title shown in the browser tab
    page_icon="🧬",                         # Favicon shown in the browser tab
    layout="centered",                      # Center the main content on the screen
    initial_sidebar_state="expanded",       # Keep the sidebar open by default
)

# ── PATH SETUP ────────────────────────────────────────────────────────────────
# Identify the root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Identify the folder containing the RAG logic
RAG_SYSTEM_PATH = os.path.join(PROJECT_ROOT, "rag_system")
# Add the RAG folder to sys.path so we can import 'med_rag' directly
if RAG_SYSTEM_PATH not in sys.path:
    sys.path.insert(0, RAG_SYSTEM_PATH)

from dotenv import load_dotenv  # Import tool to load configuration from .env
# Load environment variables (API keys, DB hosts) from the local .env file
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ── CUSTOM CSS (STYLING) ──────────────────────────────────────────────────────
# We use standard CSS to create a premium, dark-themed medical interface
st.markdown("""
<style>
  /* Import a modern sans-serif font from Google */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  /* Apply the font across the entire application */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Apply a dark gradient background to the main app container */
  .stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
  }

  /* Style the sidebar with a darker vertical gradient and a subtle border */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
  }

  /* Configure the main title banner with a vibrant green/blue gradient */
  .header-banner {
    background: linear-gradient(135deg, #1a472a 0%, #0d6e3e 50%, #1a5276 100%);
    border: 1px solid #238636;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(35, 134, 54, 0.2);
  }
  .header-banner h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
  }
  .header-banner p {
    color: #adbac7;
    font-size: 1rem;
    margin: 0;
  }

  /* Layout for the 'Answer' card shown after processing */
  .result-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #238636;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
    box-shadow: 0 4px 24px rgba(35, 134, 54, 0.12);
  }
  .result-card h3 {
    color: #3fb950;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0 0 0.8rem 0;
  }
  .answer-text {
    color: #e6edf3;
    font-size: 1rem;
    line-height: 1.75;
  }

  /* Styling for metric pills (Retrieval time, Doc count, etc.) */
  .metric-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1.2rem;
  }
  .metric-pill {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.8rem;
    color: #8b949e;
  }
  .metric-pill span {
    color: #58a6ff;
    font-weight: 600;
  }

  /* Small blue badges used to display source PMIDs */
  .pmid-badge {
    display: inline-block;
    background: #1f6feb22;
    border: 1px solid #1f6feb;
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    color: #58a6ff;
    margin: 0.2rem 0.15rem;
  }

  /* Style for the main question input area */
  .stTextArea textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
  }
  .stTextArea textarea:focus {
    border-color: #238636 !important;
    box-shadow: 0 0 0 2px rgba(35, 134, 54, 0.3) !important;
  }

  /* Style for the primary green action button */
  .stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(35, 134, 54, 0.3) !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(35, 134, 54, 0.45) !important;
  }

  /* Style for cards in the history section */
  .history-entry {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
  }
  .history-q { color: #adbac7; font-weight: 500; margin-bottom: 0.4rem; }
  .history-a { color: #79c0ff; line-height: 1.6; }

  /* Change the spinner color to match our theme green */
  .stSpinner > div { border-top-color: #2ea043 !important; }

  /* Rounded styling for dropdown menus */
  [data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
  }

  /* Custom color for slider labels */
  .stSlider label { color: #adbac7 !important; }

  /* Color for horizontal separators */
  hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)

# ── HEADER DISPLAY ────────────────────────────────────────────────────────────
# Dynamic title update based on the currently active LLM provider
llm_provider = os.getenv('LLM_PROVIDER', 'gemini').title()
if llm_provider == "Openai": llm_provider = "OpenAI"
if llm_provider == "Groq": llm_provider = "Groq"

# Render the main HTML header banner
st.markdown(f"""
<div class="header-banner">
  <h1>🧬 Medical RAG Q&amp;A System</h1>
  <p>Ask one biomedical question at a time · Powered by {llm_provider} + Elasticsearch</p>
</div>
""", unsafe_allow_html=True)

# ── SESSION STATE INITIALIZATION ──────────────────────────────────────────────
# We use st.session_state to persist data across browser refreshes
if "history" not in st.session_state:
    st.session_state.history = []  # List of previous Q&A pairs
if "rag" not in st.session_state:
    st.session_state.rag = None    # The MedRAG object (initialized on demand)
if "rag_config" not in st.session_state:
    st.session_state.rag_config = {} # Tracking if config changed to reload RAG

# ── SIDEBAR (SETTINGS PANEL) ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    # Map for human-readable labels to internal retriever IDs
    retriever_map = {
        "BM25 (Fast, lexical)": 2,
        "Hybrid (BM25 + rerank)": 3,
        "DPR – Dense Passage Retrieval": 5,
    }
    retriever_label = st.selectbox(
        "🔍 Retriever",
        list(retriever_map.keys()),
        index=0,  # Default to the most stable option
        help="BM25 is fastest; Hybrid adds neural reranking; DPR uses vector search.",
    )
    retriever_val = retriever_map[retriever_label]

    # Map for the expected medical question format (used for specific LLM prompts)
    q_type_map = {
        "Factoid (detailed answer)": 1,
        "Yes/No": 2,
        "Multiple choice (1–4)": 3,
        "List (keywords)": 4,
    }
    q_type_label = st.selectbox(
        "📋 Question Type",
        list(q_type_map.keys()),
        index=0,
        help="Adjusts system instructions based on your expected answer format.",
    )
    q_type_val = q_type_map[q_type_label]

    # Slider to control how many PubMed abstracts are retrieved per question
    n_docs = st.slider(
        "📄 Retrieved documents",
        min_value=1, max_value=15, value=5,
        help="Fewer documents result in faster answers and lower token costs.",
    )

    st.markdown("---")
    st.markdown("### 🔑 API Status")
    
    # Identify which provider is currently used for the 'Chat' functionality
    current_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    # Check if the required API key for the current provider is present in .env
    if current_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        provider_name = "Gemini"
    elif current_provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        provider_name = "Groq"
    else:
        api_key = os.getenv("OPENAI_API_KEY", "")
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        provider_name = "OpenAI"

    # Display a success/error message in the sidebar regarding the API key
    if api_key:
        st.success(f"{provider_name} key loaded ✓")
    else:
        st.error(f"{provider_name} key NOT found in .env")
    st.caption(f"Model: `{model_name}`")

    st.markdown("---")
    # Button to wipe the Q&A session history
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun() # Refresh the page immediately

    st.markdown("---")
    st.markdown(
        "<div style='color:#8b949e;font-size:0.78rem;line-height:1.6'>"
        "💡 <b>Free-tier tip:</b> Use BM25 retriever with 5 docs to minimise token usage."
        "</div>",
        unsafe_allow_html=True,
    )

    # Dynamic warning: If DPR is chosen, remind the user to build the vector index
    if retriever_label == "DPR – Dense Passage Retrieval":
        st.markdown(
            "<div style='background:#1a1f2e;border:1px solid #1f6feb;border-radius:8px;"
            "padding:0.75rem;margin-top:0.5rem;font-size:0.78rem;color:#79c0ff;line-height:1.6'>"
            "⚠️ <b>DPR one-time setup required</b><br>"
            "Run this command once to build the dense-vector index:<br>"
            "<code style='color:#e6edf3;font-size:0.75rem'>"
            "python information_retrieval/document_encoding/encode_documents_dpr.py"
            "</code></div>",
            unsafe_allow_html=True,
        )


# ── RAG INITIALIZER (CACHED) ───────────────────────────────────────────────────
# We cache the MedRAG object so it doesn't reload heavy models on every UI click
@st.cache_resource(show_spinner=False)
def load_rag(retriever_val: int, q_type_val: int, n_docs: int):
    # Import inside the function to avoid overhead during initial page load
    from med_rag import MedRAG
    return MedRAG(retriever=retriever_val, question_type=q_type_val, n_docs=n_docs)


# ── MAIN INPUT AREA ───────────────────────────────────────────────────────────
st.markdown("#### 💬 Enter your medical question")
# Text box for the user to type their biomedical inquiry
question = st.text_area(
    label="Your medical question",
    label_visibility="collapsed",
    placeholder="e.g. What are the structural proteins of a coronavirus?",
    height=110,
    key="question_input",
)

# Button to trigger the RAG pipeline
col_btn, col_tip = st.columns([1, 3])
with col_btn:
    submit = st.button("🔎 Get Answer", use_container_width=True)
with col_tip:
    # Helpful tip about rate limiting for users
    st.markdown(
        "<div style='color:#6e7681;font-size:0.82rem;padding-top:0.65rem'>"
        "One question at a time to stay within the free-tier rate limit."
        "</div>",
        unsafe_allow_html=True,
    )

# ── PROCESSING PIPELINE ───────────────────────────────────────────────────────
if submit:
    q = question.strip()
    # Validate that the input is not empty
    if not q:
        st.warning("⚠️ Please enter a question before submitting.")
    # Ensure an API key exists before attempting a call
    elif not api_key:
        st.error(f"❌ {provider_name} API key is missing. Update your `.env` file.")
    else:
        # Show a loading spinner while the background processing happens
        with st.spinner("🧠 Retrieving documents & generating answer…"):
            try:
                # 1. Initialize the MedRAG core with selected UI settings
                rag = load_rag(retriever_val, q_type_val, n_docs)
                # 2. Execute the Retrieval + Generation pipeline
                raw_json = rag.get_answer(q)
                data = json.loads(raw_json) # Parse the structured response

                # Extract individual components for UI rendering
                answer = data.get("response", "No answer returned.")
                used_pmids = data.get("used_PMIDs", [])
                retrieved_pmids = data.get("retrieved_PMIDs", [])
                ret_time = data.get("retrieval_time", 0)
                gen_time = data.get("generation_time", 0)
                error_msg = data.get("error", None)

                # Check if the LLM provider returned an explicit API error
                if error_msg:
                    st.error(f"❌ API Error: {error_msg}")
                else:
                    # ── RESULT CARD RENDERING ──────────────────────────────────
                    # Generate small badge HTML for each cited source PMID
                    pmid_html = "".join(
                        f'<span class="pmid-badge">PMID {p}</span>'
                        for p in used_pmids
                    ) if used_pmids else "<em style='color:#6e7681'>None cited</em>"

                    # Display the final answer, cited sources, and performance metrics
                    st.markdown(f"""
<div class="result-card">
  <h3>✅ Answer</h3>
  <div class="answer-text">{answer}</div>
  <hr style="border-color:#30363d;margin:1rem 0">
  <div style="font-size:0.85rem;color:#8b949e;margin-bottom:0.4rem"><b>Sources cited by model:</b></div>
  <div>{pmid_html}</div>
  <div class="metric-row">
    <div class="metric-pill">⏱ Retrieval <span>{ret_time:.2f}s</span></div>
    <div class="metric-pill">⚡ Generation <span>{gen_time:.2f}s</span></div>
    <div class="metric-pill">📄 Docs retrieved <span>{len(retrieved_pmids)}</span></div>
    <div class="metric-pill">📌 Docs cited <span>{len(used_pmids)}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

                    # 3. Save this interaction to the session state for the 'History' tab
                    st.session_state.history.insert(0, {
                        "question": q,
                        "answer": answer,
                        "used_pmids": used_pmids,
                        "ret_time": ret_time,
                        "gen_time": gen_time,
                    })

            except Exception as e:
                # Catch general errors (e.g., Elasticsearch down, SSL errors)
                st.error(f"❌ An error occurred: {e}")
                st.info("💡 Make sure Elasticsearch is running on `http://localhost:9200` and documents have been ingested.")

# ── HISTORY PANEL (LOWER SECTION) ─────────────────────────────────────────────
# If any questions have been asked, show them in a list below the main input
if st.session_state.history:
    st.markdown("---")
    st.markdown("#### 🕑 Session History")
    for idx, entry in enumerate(st.session_state.history):
        # Truncate long questions/answers for the preview
        short_q = entry["question"][:90] + "…" if len(entry["question"]) > 90 else entry["question"]
        short_a = str(entry["answer"])[:200] + "…" if len(str(entry["answer"])) > 200 else str(entry["answer"])
        pmids_str = ", ".join(str(p) for p in entry["used_pmids"]) if entry["used_pmids"] else "–"
        
        # Render a compact card for each entry in the history
        st.markdown(f"""
<div class="history-entry">
  <div class="history-q">Q{len(st.session_state.history) - idx}: {short_q}</div>
  <div class="history-a">{short_a}</div>
  <div style="margin-top:0.5rem;font-size:0.75rem;color:#6e7681">
    ⏱ {entry['ret_time']:.2f}s retrieval · ⚡ {entry['gen_time']:.2f}s gen
    &nbsp;|&nbsp; PMIDs: {pmids_str}
  </div>
</div>
""", unsafe_allow_html=True)
