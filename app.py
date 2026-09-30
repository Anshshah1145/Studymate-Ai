import os
import re
import json
import hashlib
import textwrap
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
DOCUMENTS_JSON = BASE_DIR / "documents.json"
STATS_FILE = BASE_DIR / "study_stats.json"
QUIZ_HISTORY_FILE = BASE_DIR / "quiz_history.json"

DOCUMENTS_DIR.mkdir(exist_ok=True)


# ============================================================
# ENVIRONMENT / API KEY
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        GOOGLE_API_KEY = None

if not GOOGLE_API_KEY:
    st.error(
        "GOOGLE_API_KEY is missing. Add it to your .env file "
        "or Streamlit secrets."
    )
    st.stop()


# ============================================================
# CONSTANTS
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GEMINI_MODEL = "gemini-3.5-flash-lite"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SUMMARY_BATCH_SIZE = 10

QUIZ_QUESTION_COUNT = 10
FLASHCARD_COUNT = 10


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>

    .main {
        background-color: #f8fafc;
    }

    .hero {
        padding: 45px 35px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e293b
        );
        color: white;
        margin-bottom: 30px;
    }

    .hero h1 {
        color: white !important;
        font-size: 44px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #cbd5e1 !important;
        font-size: 18px;
        line-height: 1.6;
    }

    .feature-card {
        padding: 24px;
        border-radius: 16px;
        background: #ffffff;
        color: #0f172a !important;
        border: 1px solid #e2e8f0;
        min-height: 170px;
        margin-bottom: 20px;
    }

    .feature-card h3 {
        color: #0f172a !important;
        margin-bottom: 10px;
    }

    .feature-card p {
        color: #475569 !important;
        line-height: 1.5;
    }

    .metric-card {
        padding: 20px;
        border-radius: 15px;
        background: white;
        border: 1px solid #e2e8f0;
        text-align: center;
    }

    .document-card {
        padding: 18px;
        border-radius: 15px;
        background: white;
        border: 1px solid #e2e8f0;
        margin-bottom: 12px;
    }

    .document-card h4 {
        color: #0f172a !important;
    }

    .document-card p {
        color: #475569 !important;
    }

    .answer-box {
        padding: 20px;
        border-radius: 15px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        color: #0f172a !important;
    }

    .source-box {
        padding: 12px;
        border-radius: 10px;
        background: #f1f5f9;
        color: #475569 !important;
    }

    .footer {
        text-align: center;
        padding: 30px;
        color: #64748b !important;
    }

</style>
""")


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "navigation" not in st.session_state:
    st.session_state.navigation = "Home"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_documents" not in st.session_state:
    st.session_state.selected_documents = []

if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []

if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "quiz_answer_correct" not in st.session_state:
    st.session_state.quiz_answer_correct = False

if "current_quiz_results" not in st.session_state:
    st.session_state.current_quiz_results = []

if "quiz_history_saved" not in st.session_state:
    st.session_state.quiz_history_saved = False

if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

if "flashcard_index" not in st.session_state:
    st.session_state.flashcard_index = 0

if "summary" not in st.session_state:
    st.session_state.summary = ""


# ============================================================
# NAVIGATION CALLBACK
# ============================================================

def change_page():
    st.session_state.page = st.session_state.navigation


# ============================================================
# DOCUMENT FUNCTIONS
# ============================================================

def load_documents_data():

    if not DOCUMENTS_JSON.exists():
        return []

    try:
        with open(DOCUMENTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_documents_data(data):

    with open(DOCUMENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def calculate_file_hash(file_bytes):

    return hashlib.md5(file_bytes).hexdigest()


def process_pdf(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    file_hash = calculate_file_hash(file_bytes)

    file_path = DOCUMENTS_DIR / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    loader = PyPDFLoader(str(file_path))

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source"] = uploaded_file.name
        chunk.metadata["document_id"] = file_hash

    ids = [
        f"{file_hash}_{index}"
        for index in range(len(chunks))
    ]

    vector_store = get_vector_store()

    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    document_data = load_documents_data()

    existing = [
        doc for doc in document_data
        if doc["id"] != file_hash
    ]

    existing.append({
        "id": file_hash,
        "name": uploaded_file.name,
        "pages": len(documents),
        "chunks": len(chunks)
    })

    save_documents_data(existing)

    return {
        "name": uploaded_file.name,
        "pages": len(documents),
        "chunks": len(chunks)
    }


def delete_document(document):

    vector_store = get_vector_store()

    try:

        stored_data = vector_store.get(
            where={
                "document_id": document["id"]
            }
        )

        chunk_ids = stored_data.get("ids", [])

        if chunk_ids:
            vector_store.delete(
                ids=chunk_ids
            )

    except Exception as e:
        st.warning(
            f"Could not delete vector data: {e}"
        )

    file_path = DOCUMENTS_DIR / document["name"]

    if file_path.exists():
        file_path.unlink()

    documents = load_documents_data()

    documents = [
        doc for doc in documents
        if doc["id"] != document["id"]
    ]

    save_documents_data(documents)


# ============================================================
# STATISTICS
# ============================================================

def load_stats():

    default_stats = {
        "quiz_attempts": 0,
        "quiz_questions_answered": 0,
        "quiz_correct": 0,
        "flashcards_generated": 0,
        "flashcards_reviewed": 0,
        "questions_asked": 0,
        "summaries_generated": 0
    }

    if not STATS_FILE.exists():
        return default_stats

    try:

        with open(STATS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key in default_stats:
            if key not in data:
                data[key] = default_stats[key]

        return data

    except Exception:
        return default_stats


def save_stats(stats):

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            stats,
            f,
            indent=4
        )


def update_stat(key, amount=1):

    stats = load_stats()

    stats[key] = stats.get(key, 0) + amount

    save_stats(stats)


# ============================================================
# QUIZ HISTORY
# ============================================================

def load_quiz_history():

    if not QUIZ_HISTORY_FILE.exists():
        return []

    try:

        with open(
            QUIZ_HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:
        return []


def save_quiz_history(history):

    with open(
        QUIZ_HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4,
            ensure_ascii=False
        )


def save_quiz_attempt():

    history = load_quiz_history()

    total = len(
        st.session_state.quiz_questions
    )

    score = st.session_state.quiz_score

    accuracy = (
        round(
            (score / total) * 100,
            1
        )
        if total > 0
        else 0
    )

    attempt = {

        "id": datetime.now().strftime(
            "%Y%m%d%H%M%S%f"
        ),

        "date": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "score": score,

        "total": total,

        "accuracy": accuracy,

        "documents": st.session_state.get(
            "selected_documents",
            []
        ),

        "questions":
            st.session_state.current_quiz_results.copy()
    }

    history.append(attempt)

    save_quiz_history(history)


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# VECTOR STORE
# ============================================================

@st.cache_resource
def get_vector_store():

    embeddings = get_embeddings()

    return Chroma(
        collection_name="studymate",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )


# ============================================================
# GEMINI
# ============================================================

@st.cache_resource
def get_llm():

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )


# ============================================================
# RESPONSE EXTRACTION
# ============================================================

def extract_response_content(response):

    if isinstance(
        response.content,
        str
    ):

        return response.content.strip()

    text = ""

    for block in response.content:

        if (
            isinstance(block, dict)
            and block.get("type") == "text"
        ):

            text += block.get(
                "text",
                ""
            )

    return text.strip()


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    text = text.strip()

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace(
        "```",
        ""
    ).strip()

    try:
        return json.loads(text)

    except Exception:

        match = re.search(
            r"(\[.*\])",
            text,
            re.DOTALL
        )

        if match:

            return json.loads(
                match.group(1)
            )

        match = re.search(
            r"(\{.*\})",
            text,
            re.DOTALL
        )

        if match:

            return json.loads(
                match.group(1)
            )

    raise ValueError(
        "Could not parse JSON response."
    )


# ============================================================
# FOLLOW-UP QUERY DETECTION
# ============================================================

def needs_query_rewriting(
    question,
    messages
):

    if len(messages) < 2:
        return False

    question_lower = (
        question
        .lower()
        .strip()
    )

    follow_up_phrases = [

        "what about",

        "how about",

        "tell me more about",

        "explain more about",

        "what does this mean",

        "what does that mean",

        "as mentioned above",

        "the previous one",

        "the previous answer",

        "same topic",

        "above concept"
    ]

    for phrase in follow_up_phrases:

        if phrase in question_lower:
            return True

    reference_words = [

        "it",
        "its",
        "they",
        "them",
        "this",
        "that",
        "these",
        "those",
        "their"
    ]

    words = set(
        re.findall(
            r"\b[a-zA-Z]+\b",
            question_lower
        )
    )

    return any(
        word in words
        for word in reference_words
    )


# ============================================================
# QUERY REWRITING
# ============================================================

def rewrite_query(
    question,
    messages
):

    llm = get_llm()

    conversation = ""

    for message in messages[-6:]:

        role = message["role"]

        content = message["content"]

        conversation += (
            f"{role}: {content}\n"
        )

    prompt = f"""
Rewrite the user's latest question into a
standalone question for document retrieval.

Use the conversation context only to resolve
references such as "it", "this", "that",
"they", "previous answer", etc.

Do not answer the question.

Conversation:
{conversation}

Latest question:
{question}

Return only the standalone question.
"""

    response = llm.invoke(prompt)

    return extract_response_content(
        response
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    question,
    selected_documents,
    k=3
):

    vector_store = get_vector_store()

    if selected_documents:

        results = vector_store.similarity_search(
            question,
            k=k,
            filter={
                "source": {
                    "$in": selected_documents
                }
            }
        )

    else:

        results = vector_store.similarity_search(
            question,
            k=k
        )

    return results


# ============================================================
# ASK AI
# ============================================================

def answer_question(
    question,
    selected_documents
):

    messages = st.session_state.messages

    retrieval_question = question

    if needs_query_rewriting(
        question,
        messages
    ):

        retrieval_question = rewrite_query(
            question,
            messages
        )

    results = retrieve_documents(
        retrieval_question,
        selected_documents,
        k=3
    )

    if not results:

        return (
            "I could not find relevant information "
            "in the selected documents."
        ), []

    context_parts = []

    sources = []

    for doc in results:

        page = doc.metadata.get(
            "page",
            ""
        )

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page_number = (
            int(page) + 1
            if str(page).isdigit()
            else page
        )

        context_parts.append(
            f"""
Source: {source}
Page: {page_number}

Content:
{doc.page_content}
"""
        )

        sources.append(
            f"{source} - Page {page_number}"
        )

    context = "\n".join(
        context_parts
    )

    conversation = ""

    for message in messages[-6:]:

        conversation += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    llm = get_llm()

    prompt = f"""
You are StudyMate AI, an academic study assistant.

Answer the user's question using ONLY the
provided document context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not available in the
   context, clearly say that the information
   was not found in the selected documents.
4. Explain concepts in simple student-friendly
   language.
5. Use examples only when supported by the
   documents.
6. Keep the answer organized.
7. Preserve terminology from the source material.

Conversation:
{conversation}

Document Context:
{context}

User Question:
{question}
"""

    response = llm.invoke(prompt)

    answer = extract_response_content(
        response
    )

    return answer, sources


# ============================================================
# SUMMARY GENERATION
# ============================================================

def generate_summary(
    selected_documents
):

    vector_store = get_vector_store()

    data = vector_store.get(
        where={
            "source": {
                "$in": selected_documents
            }
        },
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = data.get(
        "documents",
        []
    )

    metadatas = data.get(
        "metadatas",
        []
    )

    if not documents:

        return (
            "No content found in the selected documents."
        )

    chunks = []

    for index, content in enumerate(
        documents
    ):

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            ""
        )

        page_number = (
            int(page) + 1
            if str(page).isdigit()
            else page
        )

        chunks.append(
            f"""
Source: {source}
Page: {page_number}

{content}
"""
        )

    llm = get_llm()

    batch_summaries = []

    for i in range(
        0,
        len(chunks),
        SUMMARY_BATCH_SIZE
    ):

        batch = chunks[
            i:i + SUMMARY_BATCH_SIZE
        ]

        context = "\n".join(
            batch
        )

        prompt = f"""
You are creating study notes for an MCA student.

Summarize the following source material.

Use ONLY the provided content.

Preserve the terminology and concepts
from the source.

Include:
- Important concepts
- Definitions
- Key points
- Important explanations
- Exam-relevant points
- Formulas or examples if present

Do not add outside knowledge.

Source material:
{context}
"""

        response = llm.invoke(
            prompt
        )

        batch_summaries.append(
            extract_response_content(
                response
            )
        )

    combined = "\n\n".join(
        batch_summaries
    )

    final_prompt = f"""
Create one comprehensive study summary
from the following intermediate summaries.

Use only the provided summaries.

Organize the result into:

1. Main Topics
2. Key Concepts
3. Important Definitions
4. Important Points
5. Exam Revision Points

Keep terminology consistent with
the source material.

Intermediate summaries:
{combined}
"""

    response = llm.invoke(
        final_prompt
    )

    return extract_response_content(
        response
    )


# ============================================================
# QUIZ GENERATION
# ============================================================

def generate_quiz(
    selected_documents
):

    vector_store = get_vector_store()

    retrieval_queries = [

        "ANN architecture input layer hidden layer output layer",

        "important concepts definitions",

        "important terms explanations",

        "exam revision important points",

        "important formulas examples"
    ]

    retrieved_chunks = []

    seen_ids = set()

    for query in retrieval_queries:

        results = vector_store.similarity_search(
            query,
            k=6,
            filter={
                "source": {
                    "$in": selected_documents
                }
            }
        )

        for doc in results:

            content = doc.page_content

            key = (
                content[:200],
                doc.metadata.get("source"),
                doc.metadata.get("page")
            )

            if key not in seen_ids:

                seen_ids.add(key)

                retrieved_chunks.append(
                    doc
                )

    context_parts = []

    for doc in retrieved_chunks:

        page = doc.metadata.get(
            "page",
            ""
        )

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page_number = (
            int(page) + 1
            if str(page).isdigit()
            else page
        )

        context_parts.append(
            f"""
Source: {source}
Page: {page_number}

{doc.page_content}
"""
        )

    context = "\n".join(
        context_parts
    )

    llm = get_llm()

    prompt = f"""
Create exactly {QUIZ_QUESTION_COUNT}
multiple-choice questions from the provided
study material.

Use ONLY the source material.

Do not use outside knowledge.

Each question must have:

- question
- options with A, B, C, D
- answer
- explanation
- source

The answer must be exactly one of:
A, B, C, or D.

The source should contain the PDF name
and page number.

For ANN architecture, preserve the source
terminology:
- Input Layer
- Hidden Layer
- Output Layer

Input Layer:
takes/processes data.

Hidden Layer:
performs intermediate processing.

Output Layer:
generates final predictions.

Do not replace these with unrelated
components.

Return ONLY valid JSON.

Format:

[
  {{
    "question": "Question text",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "answer": "A",
    "explanation": "Explanation",
    "source": "PDF name - Page number"
  }}
]

Study Material:
{context}
"""

    response = llm.invoke(
        prompt
    )

    questions = extract_json(
        extract_response_content(
            response
        )
    )

    return questions[:QUIZ_QUESTION_COUNT]


# ============================================================
# FLASHCARD GENERATION
# ============================================================

def generate_flashcards(
    selected_documents
):

    vector_store = get_vector_store()

    retrieval_queries = [

        "ANN architecture input layer hidden layer output layer",

        "important concepts definitions",

        "important terms explanations",

        "exam revision important points"
    ]

    retrieved_chunks = []

    seen = set()

    for query in retrieval_queries:

        results = vector_store.similarity_search(
            query,
            k=6,
            filter={
                "source": {
                    "$in": selected_documents
                }
            }
        )

        for doc in results:

            key = (
                doc.page_content[:200],
                doc.metadata.get("source"),
                doc.metadata.get("page")
            )

            if key not in seen:

                seen.add(key)

                retrieved_chunks.append(
                    doc
                )

    context_parts = []

    for doc in retrieved_chunks:

        page = doc.metadata.get(
            "page",
            ""
        )

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page_number = (
            int(page) + 1
            if str(page).isdigit()
            else page
        )

        context_parts.append(
            f"""
Source: {source}
Page: {page_number}

{doc.page_content}
"""
        )

    context = "\n".join(
        context_parts
    )

    llm = get_llm()

    prompt = f"""
Create exactly {FLASHCARD_COUNT}
study flashcards from the source material.

Use ONLY the source material.

Do not add outside knowledge.

Each flashcard must contain:

- question
- answer
- source

For ANN architecture, use exactly
the source terminology:

Input Layer:
takes/processes data.

Hidden Layer:
performs intermediate processing.

Output Layer:
generates final predictions.

Return ONLY valid JSON.

Format:

[
  {{
    "question": "Question",
    "answer": "Answer",
    "source": "PDF name - Page number"
  }}
]

Source material:
{context}
"""

    response = llm.invoke(
        prompt
    )

    cards = extract_json(
        extract_response_content(
            response
        )
    )

    return cards[:FLASHCARD_COUNT]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🎓 StudyMate AI"
    )

    st.caption(
        "Your AI-powered study assistant"
    )

    pages = [

        "Home",

        "Ask AI",

        "Dashboard",

        "Summary",

        "Quiz",

        "Quiz History",

        "Flashcards"
    ]

    st.radio(
        "Navigation",
        pages,
        key="navigation",
        on_change=change_page
    )

    st.divider()

    st.markdown(
        "### 📄 Documents"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "Process PDFs",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning(
                "Please upload at least one PDF."
            )

        else:

            progress = st.progress(0)

            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                try:

                    process_pdf(
                        uploaded_file
                    )

                    progress.progress(
                        (index + 1)
                        / len(uploaded_files)
                    )

                except Exception as e:

                    st.error(
                        f"Error processing "
                        f"{uploaded_file.name}: {e}"
                    )

            st.success(
                "PDF processing completed."
            )

            st.rerun()

    documents = load_documents_data()

    if documents:

        document_names = [
            doc["name"]
            for doc in documents
        ]

        selected = st.multiselect(
            "Search in:",
            document_names,
            default=(
                st.session_state.selected_documents
                if st.session_state.selected_documents
                else document_names
            )
        )

        st.session_state.selected_documents = selected

    else:

        st.info(
            "Upload PDFs to start studying."
        )

    st.divider()

    st.markdown(
        "### 🗂️ Manage Documents"
    )

    if documents:

        for document in documents:

            st.markdown(
                f"""
                <div class="document-card">
                    <h4>📄 {document["name"]}</h4>
                    <p>
                        {document["pages"]} pages •
                        {document["chunks"]} chunks
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                f"Remove {document['name']}",
                key=f"remove_{document['id']}",
                use_container_width=True
            ):

                delete_document(
                    document
                )

                st.success(
                    "Document removed."
                )

                st.rerun()

    st.divider()

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    st.html("""
    <div class="hero">

        <h1>🎓 StudyMate AI</h1>

        <p>
            Your intelligent study assistant for
            college notes, PDFs, revision,
            quizzes and AI-powered learning.
        </p>

    </div>
    """)

    st.subheader(
        "Study smarter with your own documents."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.html("""
        <div class="feature-card">

            <h3>🤖 Ask AI</h3>

            <p>
                Ask questions about your uploaded
                notes and receive answers grounded
                in your study material.
            </p>

        </div>
        """)

    with col2:

        st.html("""
        <div class="feature-card">

            <h3>📝 AI Quiz</h3>

            <p>
                Generate multiple-choice questions
                automatically from your notes and
                test your preparation.
            </p>

        </div>
        """)

    with col3:

        st.html("""
        <div class="feature-card">

            <h3>📚 Flashcards</h3>

            <p>
                Quickly revise important concepts,
                definitions and exam topics with
                AI-generated flashcards.
            </p>

        </div>
        """)

    col4, col5, col6 = st.columns(3)

    with col4:

        st.html("""
        <div class="feature-card">

            <h3>📖 Summary</h3>

            <p>
                Generate structured study notes
                from your uploaded PDFs.
            </p>

        </div>
        """)

    with col5:

        st.html("""
        <div class="feature-card">

            <h3>📊 Dashboard</h3>

            <p>
                Track your questions, quizzes,
                flashcards and overall study
                activity.
            </p>

        </div>
        """)

    with col6:

        st.html("""
        <div class="feature-card">

            <h3>📚 Quiz History</h3>

            <p>
                Review all previous quiz attempts,
                correct answers and questions
                that need revision.
            </p>

        </div>
        """)


# ============================================================
# ASK AI
# ============================================================

elif st.session_state.page == "Ask AI":

    st.title(
        "🤖 Ask StudyMate AI"
    )

    if not st.session_state.selected_documents:

        st.warning(
            "Please upload and select at least one PDF."
        )

    else:

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

                if (
                    message["role"] == "assistant"
                    and message.get("sources")
                ):

                    st.caption(
                        "Sources: "
                        + " | ".join(
                            message["sources"]
                        )
                    )

        question = st.chat_input(
            "Ask something from your notes..."
        )

        if question:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):

                st.markdown(
                    question
                )

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Thinking..."
                ):

                    try:

                        answer, sources = answer_question(
                            question,
                            st.session_state.selected_documents
                        )

                        st.markdown(
                            answer
                        )

                        if sources:

                            st.caption(
                                "Sources: "
                                + " | ".join(
                                    sources
                                )
                            )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            }
                        )

                        update_stat(
                            "questions_asked"
                        )

                    except Exception as e:

                        answer = (
                            "An error occurred: "
                            f"{e}"
                        )

                        st.error(
                            answer
                        )


# ============================================================
# DASHBOARD
# ============================================================

elif st.session_state.page == "Dashboard":

    st.title(
        "📊 Study Dashboard"
    )

    stats = load_stats()

    documents = load_documents_data()

    total_questions = stats[
        "quiz_questions_answered"
    ]

    correct = stats[
        "quiz_correct"
    ]

    accuracy = (
        round(
            correct / total_questions * 100,
            1
        )
        if total_questions > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📄 Documents",
            len(documents)
        )

    with col2:

        st.metric(
            "❓ Questions Asked",
            stats["questions_asked"]
        )

    with col3:

        st.metric(
            "📝 Quiz Attempts",
            stats["quiz_attempts"]
        )

    with col4:

        st.metric(
            "🎯 Quiz Accuracy",
            f"{accuracy}%"
        )

    st.divider()

    col5, col6, col7 = st.columns(3)

    with col5:

        st.metric(
            "Quiz Questions",
            stats["quiz_questions_answered"]
        )

    with col6:

        st.metric(
            "Flashcards Generated",
            stats["flashcards_generated"]
        )

    with col7:

        st.metric(
            "Flashcards Reviewed",
            stats["flashcards_reviewed"]
        )

    st.divider()

    st.subheader(
        "📚 Documents"
    )

    for document in documents:

        st.write(
            f"📄 **{document['name']}** — "
            f"{document['pages']} pages • "
            f"{document['chunks']} chunks"
        )


# ============================================================
# SUMMARY
# ============================================================

elif st.session_state.page == "Summary":

    st.title(
        "📖 Study Summary"
    )

    documents = load_documents_data()

    if not documents:

        st.warning(
            "Upload PDFs first."
        )

    else:

        if not st.session_state.selected_documents:

            st.warning(
                "Select at least one document."
            )

        else:

            if st.button(
                "✨ Generate Summary",
                use_container_width=True
            ):

                with st.spinner(
                    "Generating study summary..."
                ):

                    try:

                        st.session_state.summary = (
                            generate_summary(
                                st.session_state.selected_documents
                            )
                        )

                        update_stat(
                            "summaries_generated"
                        )

                    except Exception as e:

                        st.error(
                            f"Error generating summary: {e}"
                        )

            if st.session_state.summary:

                st.markdown(
                    st.session_state.summary
                )


# ============================================================
# QUIZ
# ============================================================

elif st.session_state.page == "Quiz":

    st.title(
        "📝 AI Quiz"
    )

    if not st.session_state.selected_documents:

        st.warning(
            "Please select at least one document."
        )

    else:

        if not st.session_state.quiz_questions:

            st.write(
                "Generate a quiz from your selected study material."
            )

            if st.button(
                "🎯 Generate Quiz",
                use_container_width=True
            ):

                with st.spinner(
                    "Generating quiz..."
                ):

                    try:

                        questions = generate_quiz(
                            st.session_state.selected_documents
                        )

                        st.session_state.quiz_questions = (
                            questions
                        )

                        st.session_state.quiz_index = 0

                        st.session_state.quiz_score = 0

                        st.session_state.quiz_submitted = False

                        st.session_state.quiz_answer_correct = False

                        st.session_state.current_quiz_results = []

                        st.session_state.quiz_history_saved = False

                        update_stat(
                            "quiz_attempts"
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Error generating quiz: {e}"
                        )

        else:

            questions = (
                st.session_state.quiz_questions
            )

            current_index = (
                st.session_state.quiz_index
            )

            total_questions = len(
                questions
            )

            # ----------------------------------------
            # FINAL SCORE
            # ----------------------------------------

            if current_index >= total_questions:

                score = (
                    st.session_state.quiz_score
                )

                percentage = (
                    round(
                        score /
                        total_questions *
                        100,
                        1
                    )
                    if total_questions > 0
                    else 0
                )

                st.success(
                    f"🎉 Quiz Completed!"
                )

                st.metric(
                    "Final Score",
                    f"{score}/{total_questions}"
                )

                st.metric(
                    "Accuracy",
                    f"{percentage}%"
                )

                st.divider()

                st.subheader(
                    "📚 Review"
                )

                for index, result in enumerate(
                    st.session_state.current_quiz_results,
                    start=1
                ):

                    if result["is_correct"]:

                        st.success(
                            f"Question {index}: Correct"
                        )

                    else:

                        st.error(
                            f"Question {index}: Incorrect"
                        )

                    st.write(
                        f"**{result['question']}**"
                    )

                    st.write(
                        "Your answer: "
                        f"**{result['selected_answer']}. "
                        f"{result['selected_text']}**"
                    )

                    if not result["is_correct"]:

                        st.write(
                            "Correct answer: "
                            f"**{result['correct_answer']}. "
                            f"{result['correct_text']}**"
                        )

                    if result.get(
                        "explanation"
                    ):

                        st.info(
                            result["explanation"]
                        )

                    if result.get("source"):

                        st.caption(
                            f"📚 {result['source']}"
                        )

                    st.divider()

                if st.button(
                    "🔄 Start New Quiz",
                    use_container_width=True
                ):

                    st.session_state.quiz_questions = []

                    st.session_state.quiz_index = 0

                    st.session_state.quiz_score = 0

                    st.session_state.quiz_submitted = False

                    st.session_state.quiz_answer_correct = False

                    st.session_state.current_quiz_results = []

                    st.session_state.quiz_history_saved = False

                    st.rerun()

            else:

                question = questions[
                    current_index
                ]

                st.progress(
                    (current_index + 1)
                    / total_questions
                )

                st.write(
                    f"### Question "
                    f"{current_index + 1} "
                    f"of {total_questions}"
                )

                st.markdown(
                    f"## {question['question']}"
                )

                options = question[
                    "options"
                ]

                option_texts = [

                    f"A. {options['A']}",

                    f"B. {options['B']}",

                    f"C. {options['C']}",

                    f"D. {options['D']}"
                ]

                selected_option = st.radio(
                    "Select your answer:",
                    option_texts,
                    key=f"quiz_option_{current_index}",
                    disabled=st.session_state.quiz_submitted
                )

                selected_answer = selected_option[
                    0
                ]

                # ----------------------------------------
                # SUBMIT ANSWER
                # ----------------------------------------

                if not st.session_state.quiz_submitted:

                    if st.button(
                        "Submit Answer",
                        use_container_width=True
                    ):

                        is_correct = (
                            selected_answer
                            == question["answer"]
                        )

                        if is_correct:

                            st.session_state.quiz_score += 1

                            st.session_state.quiz_answer_correct = True

                            st.success(
                                "✅ Correct!"
                            )

                        else:

                            st.session_state.quiz_answer_correct = False

                            st.error(
                                "❌ Incorrect!"
                            )

                        result = {

                            "question":
                                question["question"],

                            "selected_answer":
                                selected_answer,

                            "selected_text":
                                question["options"][
                                    selected_answer
                                ],

                            "correct_answer":
                                question["answer"],

                            "correct_text":
                                question["options"][
                                    question["answer"]
                                ],

                            "is_correct":
                                is_correct,

                            "explanation":
                                question.get(
                                    "explanation",
                                    ""
                                ),

                            "source":
                                question.get(
                                    "source",
                                    ""
                                )
                        }

                        st.session_state.current_quiz_results.append(
                            result
                        )

                        st.session_state.quiz_submitted = True

                        update_stat(
                            "quiz_questions_answered"
                        )

                        if is_correct:

                            update_stat(
                                "quiz_correct"
                            )

                        # Save completed attempt
                        if (
                            current_index
                            == total_questions - 1
                        ):

                            if not st.session_state.quiz_history_saved:

                                save_quiz_attempt()

                                st.session_state.quiz_history_saved = True

                        st.rerun()

                # ----------------------------------------
                # RESULT
                # ----------------------------------------

                else:

                    if st.session_state.quiz_answer_correct:

                        st.success(
                            "✅ Correct answer!"
                        )

                    else:

                        st.error(
                            "❌ Incorrect answer."
                        )

                        st.write(
                            "Correct answer:"
                        )

                        st.write(
                            f"**{question['answer']}. "
                            f"{question['options'][question['answer']]}**"
                        )

                    if question.get(
                        "explanation"
                    ):

                        st.info(
                            f"💡 {question['explanation']}"
                        )

                    if question.get(
                        "source"
                    ):

                        st.caption(
                            f"📚 Source: "
                            f"{question['source']}"
                        )

                    st.divider()

                    if st.button(
                        (
                            "Next Question →"
                            if current_index
                            < total_questions - 1
                            else "View Final Result"
                        ),
                        use_container_width=True
                    ):

                        st.session_state.quiz_index += 1

                        st.session_state.quiz_submitted = False

                        st.session_state.quiz_answer_correct = False

                        st.rerun()


# ============================================================
# QUIZ HISTORY
# ============================================================

elif st.session_state.page == "Quiz History":

    st.title(
        "📚 Quiz History"
    )

    st.write(
        "Review all your previous quiz attempts, "
        "correct answers and incorrect answers."
    )

    history = load_quiz_history()

    if not history:

        st.info(
            "No quiz history yet. Complete a quiz "
            "to see your results here."
        )

    else:

        # ============================================
        # OVERALL STATISTICS
        # ============================================

        total_attempts = len(
            history
        )

        total_questions = sum(
            len(
                attempt.get(
                    "questions",
                    []
                )
            )
            for attempt in history
        )

        total_correct = sum(

            sum(

                1

                for question in attempt.get(
                    "questions",
                    []
                )

                if question.get(
                    "is_correct",
                    False
                )

            )

            for attempt in history
        )

        total_incorrect = (
            total_questions
            - total_correct
        )

        overall_accuracy = (

            round(
                total_correct
                / total_questions
                * 100,
                1
            )

            if total_questions > 0

            else 0
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:

            st.metric(
                "Attempts",
                total_attempts
            )

        with col2:

            st.metric(
                "Questions",
                total_questions
            )

        with col3:

            st.metric(
                "Correct",
                total_correct
            )

        with col4:

            st.metric(
                "Incorrect",
                total_incorrect
            )

        with col5:

            st.metric(
                "Accuracy",
                f"{overall_accuracy}%"
            )

        st.divider()

        # ============================================
        # TABS
        # ============================================

        tab1, tab2, tab3 = st.tabs(
            [
                "📊 All Attempts",
                "✅ Correct Questions",
                "❌ Incorrect Questions"
            ]
        )

        # ============================================
        # ALL ATTEMPTS
        # ============================================

        with tab1:

            for attempt_number, attempt in enumerate(
                reversed(history),
                start=1
            ):

                score = attempt.get(
                    "score",
                    0
                )

                total = attempt.get(
                    "total",
                    0
                )

                accuracy = attempt.get(
                    "accuracy",
                    0
                )

                date = attempt.get(
                    "date",
                    "Unknown"
                )

                original_number = (
                    total_attempts
                    - attempt_number
                    + 1
                )

                with st.expander(
                    f"📝 Quiz Attempt "
                    f"#{original_number} — "
                    f"{score}/{total} "
                    f"({accuracy}%) — "
                    f"{date}"
                ):

                    questions = attempt.get(
                        "questions",
                        []
                    )

                    for index, question in enumerate(
                        questions,
                        start=1
                    ):

                        if question.get(
                            "is_correct",
                            False
                        ):

                            st.success(
                                f"Question {index}: Correct"
                            )

                        else:

                            st.error(
                                f"Question {index}: Incorrect"
                            )

                        st.write(
                            f"**Q{index}. "
                            f"{question.get('question', '')}**"
                        )

                        st.write(
                            "Your answer: "
                            f"**{question.get('selected_answer', '')}. "
                            f"{question.get('selected_text', '')}**"
                        )

                        if not question.get(
                            "is_correct",
                            False
                        ):

                            st.write(
                                "Correct answer: "
                                f"**{question.get('correct_answer', '')}. "
                                f"{question.get('correct_text', '')}**"
                            )

                        if question.get(
                            "explanation"
                        ):

                            st.info(
                                f"💡 "
                                f"{question.get('explanation')}"
                            )

                        if question.get(
                            "source"
                        ):

                            st.caption(
                                f"📚 Source: "
                                f"{question.get('source')}"
                            )

                        st.divider()

        # ============================================
        # CORRECT QUESTIONS
        # ============================================

        with tab2:

            correct_found = False

            for attempt_number, attempt in enumerate(
                history,
                start=1
            ):

                for question_number, question in enumerate(
                    attempt.get(
                        "questions",
                        []
                    ),
                    start=1
                ):

                    if question.get(
                        "is_correct",
                        False
                    ):

                        correct_found = True

                        st.success(
                            f"✅ Correct — "
                            f"Quiz #{attempt_number}, "
                            f"Question #{question_number}"
                        )

                        st.write(
                            f"**{question.get('question', '')}**"
                        )

                        st.write(
                            "Your answer: "
                            f"**{question.get('selected_answer', '')}. "
                            f"{question.get('selected_text', '')}**"
                        )

                        if question.get(
                            "explanation"
                        ):

                            st.info(
                                f"💡 "
                                f"{question.get('explanation')}"
                            )

                        if question.get(
                            "source"
                        ):

                            st.caption(
                                f"📚 Source: "
                                f"{question.get('source')}"
                            )

                        st.divider()

            if not correct_found:

                st.info(
                    "No correct answers recorded yet."
                )

        # ============================================
        # INCORRECT QUESTIONS
        # ============================================

        with tab3:

            incorrect_found = False

            for attempt_number, attempt in enumerate(
                history,
                start=1
            ):

                for question_number, question in enumerate(
                    attempt.get(
                        "questions",
                        []
                    ),
                    start=1
                ):

                    if not question.get(
                        "is_correct",
                        False
                    ):

                        incorrect_found = True

                        st.error(
                            f"❌ Incorrect — "
                            f"Quiz #{attempt_number}, "
                            f"Question #{question_number}"
                        )

                        st.write(
                            f"**{question.get('question', '')}**"
                        )

                        st.write(
                            "Your answer: "
                            f"**{question.get('selected_answer', '')}. "
                            f"{question.get('selected_text', '')}**"
                        )

                        st.write(
                            "Correct answer: "
                            f"**{question.get('correct_answer', '')}. "
                            f"{question.get('correct_text', '')}**"
                        )

                        if question.get(
                            "explanation"
                        ):

                            st.info(
                                f"💡 "
                                f"{question.get('explanation')}"
                            )

                        if question.get(
                            "source"
                        ):

                            st.caption(
                                f"📚 Source: "
                                f"{question.get('source')}"
                            )

                        st.divider()

            if not incorrect_found:

                st.success(
                    "🎉 No incorrect answers recorded yet!"
                )


# ============================================================
# FLASHCARDS
# ============================================================

elif st.session_state.page == "Flashcards":

    st.title(
        "🗂️ Flashcards"
    )

    if not st.session_state.selected_documents:

        st.warning(
            "Please select at least one document."
        )

    else:

        if not st.session_state.flashcards:

            st.write(
                "Generate AI-powered flashcards "
                "from your study material."
            )

            if st.button(
                "✨ Generate Flashcards",
                use_container_width=True
            ):

                with st.spinner(
                    "Generating flashcards..."
                ):

                    try:

                        cards = generate_flashcards(
                            st.session_state.selected_documents
                        )

                        st.session_state.flashcards = cards

                        st.session_state.flashcard_index = 0

                        update_stat(
                            "flashcards_generated",
                            len(cards)
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Error generating flashcards: {e}"
                        )

        else:

            cards = (
                st.session_state.flashcards
            )

            index = (
                st.session_state.flashcard_index
            )

            if index >= len(cards):

                st.success(
                    "🎉 You reviewed all flashcards!"
                )

                if st.button(
                    "🔄 Restart Flashcards",
                    use_container_width=True
                ):

                    st.session_state.flashcard_index = 0

                    st.rerun()

            else:

                card = cards[index]

                st.progress(
                    (index + 1)
                    / len(cards)
                )

                st.write(
                    f"### Card {index + 1} "
                    f"of {len(cards)}"
                )

                st.markdown(
                    f"## ❓ {card['question']}"
                )

                st.divider()

                st.markdown(
                    f"### 💡 Answer"
                )

                st.write(
                    card["answer"]
                )

                if card.get("source"):

                    st.caption(
                        f"📚 Source: {card['source']}"
                    )

                st.divider()

                col1, col2, col3 = st.columns(3)

                with col1:

                    if st.button(
                        "← Previous",
                        disabled=index == 0,
                        use_container_width=True
                    ):

                        st.session_state.flashcard_index -= 1

                        st.rerun()

                with col2:

                    if st.button(
                        "🔄 Restart",
                        use_container_width=True
                    ):

                        st.session_state.flashcard_index = 0

                        st.rerun()

                with col3:

                    if st.button(
                        "Next →",
                        use_container_width=True
                    ):

                        st.session_state.flashcard_index += 1

                        update_stat(
                            "flashcards_reviewed"
                        )

                        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    <p>
        🎓 StudyMate AI
    </p>

    <p>
        AI-powered learning from your own study material.
    </p>

</div>
""")