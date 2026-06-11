import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page setup for modern SaaS-style layout
st.set_page_config(
    page_title="CodeAlpha FAQ AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load FAQ dataset once
@st.cache_data(show_spinner=False)
def load_faq_data(path: str = "faq_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Question", "Answer"]).reset_index(drop=True)
    return df

faq_data = load_faq_data()
faq_questions = faq_data["Question"].astype(str).tolist()
faq_answers = faq_data["Answer"].astype(str).tolist()

# Build the vectorizer once and cache the vector matrix
@st.cache_data(show_spinner=False)
def build_vectorizer(corpus):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(corpus)
    return vectorizer, vectors

vectorizer, question_vectors = build_vectorizer(faq_questions)

# Initialize session state for persistent experience
def init_session_state():
    default_values = {
        "chat_history": [],
        "query_count": 0,
        "last_question": "None yet",
        "accuracy_score": 0.0,
        "confidence_threshold": 0.38,
        "search_query": "",
        "search_results": [],
        "suggested_questions": faq_questions[:5],
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Apply custom CSS for a professional dark theme
def apply_custom_theme():
    css = """
    <style>
    :root {
        color-scheme: dark;
        font-family: Inter, system-ui, sans-serif;
    }
    .css-18ni7ap.e8zbici2 {
        background: linear-gradient(135deg, #0b1120 0%, #131f36 40%, #1f2b4f 100%);
    }
    .stApp {
        background: radial-gradient(circle at top left, rgba(59, 130, 246, 0.14), transparent 28%),
                    radial-gradient(circle at bottom right, rgba(16, 185, 129, 0.12), transparent 32%),
                    #090d18;
        color: #e5e9f0;
    }
    .block-container {
        padding: 2rem 2rem 3rem 2rem;
        max-width: 1440px;
    }
    .hero-card, .glass-card, .stat-card, .chat-card, .faq-card {
        background: rgba(18, 26, 46, 0.76);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.18);
        backdrop-filter: blur(18px);
        border-radius: 26px;
    }
    .hero-card {
        padding: 2.2rem;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        padding: 1.5rem;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.34);
    }
    .text-gradient {
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .button-gradient {
        background: linear-gradient(135deg, #6366f1, #22d3ee);
        color: #ffffff;
        border: none;
        border-radius: 999px;
        padding: 0.95rem 1.8rem;
        font-weight: 600;
        cursor: pointer;
    }
    .button-gradient:hover {
        opacity: 0.95;
    }
    .chat-card {
        padding: 1.5rem;
        min-height: 520px;
    }
    .message-user, .message-bot {
        padding: 1rem 1.2rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        max-width: 92%;
        line-height: 1.65;
    }
    .message-user {
        background: rgba(59, 130, 246, 0.12);
        color: #dbeafe;
        align-self: flex-end;
        border: 1px solid rgba(59, 130, 246, 0.22);
    }
    .message-bot {
        background: rgba(79, 70, 229, 0.14);
        color: #f8fafc;
        border: 1px solid rgba(99, 102, 241, 0.18);
    }
    .message-row {
        display: flex;
        flex-direction: column;
    }
    .chip {
        display: inline-flex;
        background: rgba(255,255,255,0.06);
        color: #cbd5e1;
        padding: 0.55rem 0.95rem;
        margin: 0.2rem 0.2rem 0.2rem 0;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        font-size: 0.92rem;
    }
    .faq-card h4, .glass-card h4, .stat-card h4 {
        color: #ffffff;
    }
    .footer {
        color: #94a3b8;
        font-size: 0.95rem;
        text-align: center;
        padding-top: 1.5rem;
    }
    .footer a {
        color: #60a5fa;
        text-decoration: none;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_custom_theme()

# Utility functions for search and response

def compute_similarity(query: str):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, question_vectors).flatten()
    ranked_indices = scores.argsort()[::-1]
    return scores, ranked_indices


def get_best_answer(query: str, threshold: float):
    scores, ranked_indices = compute_similarity(query)
    best_idx = int(ranked_indices[0])
    best_score = float(scores[best_idx])
    matched_question = faq_data.loc[best_idx, "Question"]
    matched_answer = faq_data.loc[best_idx, "Answer"]
    similar_questions = []
    for idx in ranked_indices[1:5]:
        similar_questions.append(
            {
                "question": faq_data.loc[int(idx), "Question"],
                "score": float(scores[int(idx)]),
            }
        )
    no_match = best_score < threshold
    return {
        "index": best_idx,
        "score": best_score,
        "matched_question": matched_question,
        "matched_answer": matched_answer,
        "similar_questions": similar_questions,
        "no_match": no_match,
    }


def update_statistics(new_score: float):
    st.session_state.query_count += 1
    st.session_state.last_question = st.session_state.user_question
    matched_queries = [entry["score"] for entry in st.session_state.chat_history if entry["score"] >= st.session_state.confidence_threshold]
    if st.session_state.chat_history:
        correct = sum(1 for entry in st.session_state.chat_history if entry["score"] >= st.session_state.confidence_threshold)
        st.session_state.accuracy_score = (correct / st.session_state.query_count) * 100
    else:
        st.session_state.accuracy_score = 0.0


def reset_chat():
    st.session_state.chat_history = []
    st.session_state.query_count = 0
    st.session_state.last_question = "None yet"
    st.session_state.accuracy_score = 0.0
    st.session_state.search_query = ""
    st.session_state.search_results = []

# Main page layout
with st.container():
    st.markdown(
        """
        <div class='hero-card'>
            <div style='display:flex; flex-wrap:wrap; justify-content:space-between; gap:1.5rem;'>
                <div style='max-width: 640px;'>
                    <p style='color:#60a5fa; font-weight:700; letter-spacing:0.18em; text-transform:uppercase; margin-bottom:0.75rem;'>CodeAlpha AI Assistant</p>
                    <h1 style='font-size:clamp(2.6rem, 4vw, 3.6rem); margin-bottom:1rem; line-height:1.05;'>Professional FAQ Chatbot for modern SaaS portfolios</h1>
                    <p style='color:#cbd5e1; font-size:1rem; line-height:1.8;'>Ask real questions, explore suggested FAQ prompts, and receive accurate, fast responses powered by TF-IDF and cosine similarity.</p>
                    <div style='display:flex; flex-wrap:wrap; gap:0.75rem; margin-top:1.5rem;'>
                        <span class='chip'>Dark UI</span>
                        <span class='chip'>NLP Matching</span>
                        <span class='chip'>FAQ Dashboard</span>
                        <span class='chip'>Session History</span>
                    </div>
                </div>
                <div style='min-width:280px; flex:1;'>
                    <div class='glass-card' style='padding:1.5rem;'>
                        <p style='margin:0; color:#a5b4fc; letter-spacing:0.12em; text-transform:uppercase; font-size:0.85rem;'>Live performance snapshot</p>
                        <div style='display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:1rem; margin-top:1.5rem;'>
                            <div class='stat-card'>
                                <h4>Total FAQs</h4>
                                <p style='font-size:2rem; margin:0.35rem 0 0; font-weight:700;'>""" + str(len(faq_data)) + """</p>
                            </div>
                            <div class='stat-card'>
                                <h4>User Queries</h4>
                                <p style='font-size:2rem; margin:0.35rem 0 0; font-weight:700;'>""" + str(st.session_state.query_count) + """</p>
                            </div>
                            <div class='stat-card'>
                                <h4>Accuracy</h4>
                                <p style='font-size:2rem; margin:0.35rem 0 0; font-weight:700;'>""" + f"{st.session_state.accuracy_score:.1f}%" + """</p>
                            </div>
                            <div class='stat-card'>
                                <h4>Recent Question</h4>
                                <p style='font-size:1rem; margin:0.35rem 0 0; font-weight:600; color:#d1d5db;'>""" + st.session_state.last_question + """</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Chat and dashboard columns
chat_col, sidebar_col = st.columns([2.2, 1])

with chat_col:
    st.markdown("<div class='glass-card chat-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin-top:0; color:#fff;'>Ask anything from the FAQ</h2>", unsafe_allow_html=True)
    st.write("Use the chat box below to ask questions about the dataset. The chatbot returns the best FAQ answer and related question recommendations.")

    with st.form(key="chat_form"):
        user_question = st.text_area("Your question", key="user_question", height=120, placeholder="e.g. What is CodeAlpha?" )
        cols = st.columns([0.6, 0.4])
        with cols[0]:
            submit_button = st.form_submit_button("Send question")
        with cols[1]:
            clear_button = st.form_submit_button("Clear chat")

    if clear_button:
        reset_chat()

    if submit_button:
        user_query = user_question.strip()
        if user_query == "":
            st.warning("Please enter a question before sending.")
        else:
            with st.spinner("Analyzing your question and matching the most relevant FAQ..."):
                result = get_best_answer(user_query, st.session_state.confidence_threshold)
                entry = {
                    "question": user_query,
                    "answer": result["matched_answer"],
                    "matched_question": result["matched_question"],
                    "score": result["score"],
                    "no_match": result["no_match"],
                    "recommendations": result["similar_questions"],
                }
                st.session_state.chat_history.append(entry)
                update_statistics(result["score"])

    if st.session_state.chat_history:
        for item in reversed(st.session_state.chat_history[-8:]):
            message_class = "message-user" if item["question"] else "message-bot"
            st.markdown(
                f"<div class='message-row'>"
                f"<div class='message-user'><strong>You:</strong> {item['question']}</div>"
                f"<div class='message-bot'><strong>Answer:</strong> {item['answer']}<br><small style='opacity:0.75;'>Matched FAQ: {item['matched_question']} • Confidence: {item['score']*100:.1f}%</small></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if item["recommendations"]:
                suggestions_html = "".join(
                    f"<span class='chip'>{rec['question']} ({rec['score']*100:.0f}%)</span>"
                    for rec in item["recommendations"]
                )
                st.markdown(
                    f"<div style='margin-bottom:1rem;'>"
                    f"<strong style='color:#cbd5e1;'>Similar questions:</strong> {suggestions_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Start the conversation with a question from the FAQ dataset.")

    if st.session_state.chat_history:
        last_entry = st.session_state.chat_history[-1]
        if last_entry["no_match"]:
            st.warning(
                "I found the closest FAQ but my confidence is low. Please try rephrasing the question or browse the FAQ section for more precise help."
            )

    st.markdown("</div>", unsafe_allow_html=True)

with sidebar_col:
    st.markdown("<div class='glass-card' style='padding:1.5rem;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0; color:#fff;'>Dashboard & FAQ tools</h3>", unsafe_allow_html=True)
    st.metric(label="FAQ Count", value=len(faq_data))
    st.metric(label="Queries", value=st.session_state.query_count)
    st.metric(label="Accuracy", value=f"{st.session_state.accuracy_score:.1f}%")
    st.markdown("<hr style='border-color:rgba(148,163,184,0.12);'>", unsafe_allow_html=True)
    st.text_input("Search FAQ library", key="search_query", placeholder="Type keywords and search...", label_visibility="hidden")
    if st.button("Search FAQs", key="search_button"):
        query = st.session_state.search_query.strip()
        if query:
            with st.spinner("Searching related FAQ questions..."):
                scores, ranked = compute_similarity(query)
                st.session_state.search_results = [
                    {
                        "question": faq_data.loc[int(idx), "Question"],
                        "answer": faq_data.loc[int(idx), "Answer"],
                        "score": float(scores[int(idx)]),
                    }
                    for idx in ranked[:6]
                ]
        else:
            st.session_state.search_results = []
    if st.session_state.search_results:
        for result in st.session_state.search_results:
            st.markdown(
                f"<div class='faq-card' style='padding:1rem; margin-bottom:0.85rem;'>"
                f"<strong>{result['question']}</strong>"
                f"<p style='margin:0.45rem 0 0; color:#cbd5e1;'>{result['answer']}</p>"
                f"<p style='margin:0.5rem 0 0; font-size:0.9rem; color:#93c5fd;'>Confidence {result['score']*100:.0f}%</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.write("Search the FAQ dataset for keywords and phrases.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card' style='padding:1.5rem; margin-top:1.25rem;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0; color:#fff;'>Suggested questions</h3>", unsafe_allow_html=True)
    for suggested in st.session_state.suggested_questions:
        st.markdown(f"<div class='chip'>{suggested}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card' style='padding:1.5rem; margin-top:1.25rem;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0; color:#fff;'>Performance controls</h3>", unsafe_allow_html=True)
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_threshold,
        step=0.01,
    )
    st.session_state.confidence_threshold = threshold
    st.markdown(
        f"<p style='color:#cbd5e1;'>FAQ answers are flagged as low confidence below <strong>{threshold*100:.0f}%</strong>.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# FAQ library with expandable sections
st.markdown("<div class='glass-card' style='padding:1.8rem; margin-top:1.5rem;'>", unsafe_allow_html=True)
st.markdown("<h2 style='margin-top:0; color:#fff;'>Expanded FAQ section</h2>", unsafe_allow_html=True)
st.write("Browse the full FAQ dataset and explore answers to frequently asked questions.")

with st.expander("View full FAQ dataset", expanded=True):
    for idx, row in faq_data.iterrows():
        st.markdown(
            f"<div class='faq-card' style='padding:1rem; margin-bottom:0.95rem;'>"
            f"<strong>{row['Question']}</strong>"
            f"<p style='margin:0.55rem 0 0; color:#cbd5e1;'>{row['Answer']}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<div class='footer'>"
    "Built with Streamlit, Pandas, and scikit-learn · Developed by CodeAlpha"  
    "</div>",
    unsafe_allow_html=True,
)
