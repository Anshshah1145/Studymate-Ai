# 🎓 StudyMate AI

StudyMate AI is an AI-powered study assistant that uses **Retrieval-Augmented Generation (RAG)** to help students learn from their own PDF study materials.

Users can upload college notes and ask questions, generate summaries, take AI-generated quizzes, review quiz history, and revise using flashcards.

---

## ✨ Features

### 🤖 Ask AI

- Ask questions from uploaded PDFs
- RAG-based question answering
- Context-aware follow-up questions
- Source PDF and page references

### 📖 AI Summary

Generate structured study notes containing:

- Main topics
- Key concepts
- Important definitions
- Important points
- Exam revision points

### 📝 AI Quiz

Generate multiple-choice questions from study materials.

- 10-question quizzes
- Instant answer checking
- Score and accuracy
- Explanations
- Source references

### 📚 Quiz History

Review previous quiz attempts.

- Total attempts
- Total questions
- Correct answers
- Incorrect answers
- Overall accuracy
- Previous scores
- Correct questions
- Incorrect questions
- Selected answer
- Correct answer
- Explanation
- Source

### 🗂️ Flashcards

Generate AI-powered flashcards for quick revision.

### 📊 Study Dashboard

Track:

- Documents
- Questions asked
- Quiz attempts
- Quiz accuracy
- Flashcards generated
- Flashcards reviewed

### 📄 Multiple PDF Support

- Upload multiple PDFs
- Select specific documents for searching
- Process documents
- Remove documents

---

## 🧠 RAG Architecture

```text
PDF
 │
 ▼
PDF Loader
 │
 ▼
Text Chunking
 │
 ▼
Hugging Face Embeddings
 │
 ▼
ChromaDB
 │
 ▼
Similarity Search
 │
 ▼
Relevant Context
 │
 ▼
Gemini
 │
 ▼
AI Answer + Sources
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Application development |
| Streamlit | Web interface |
| LangChain | RAG pipeline |
| ChromaDB | Vector database |
| Hugging Face | Embeddings |
| Sentence Transformers | Text embeddings |
| Gemini | Large Language Model |
| PyPDF | PDF processing |
| python-dotenv | Environment variables |

---

## 📁 Project Structure

```text
Studymate-Ai/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── src/
```

### Files excluded from GitHub

The following files and folders contain local, generated, or sensitive data and should not be uploaded:

```text
.env
venv/
chroma_db/
documents/
documents.json
quiz_history.json
study_stats.json
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Anshshah1145/Studymate-Ai.git
```

### 2. Open the Project

```bash
cd Studymate-Ai
```

### 3. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 4. Activate the Virtual Environment

#### macOS / Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Key Setup

StudyMate AI uses the **Gemini API**.

Create a file named:

```text
.env
```

Add your API key:

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

Replace `YOUR_GEMINI_API_KEY` with your own Gemini API key.

> ⚠️ Never upload your `.env` file or API key to GitHub.

---

## ▶️ Run the Application

After installing the dependencies, run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔄 How It Works

### 1. Upload PDF

The user uploads one or more study PDFs.

### 2. Process Documents

StudyMate AI processes the documents through the following pipeline:

```text
PDF
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
Embeddings
 ↓
ChromaDB
```

### 3. Ask a Question

The user asks a question about their study material.

### 4. Retrieve Relevant Information

ChromaDB performs similarity search and retrieves the most relevant document chunks.

### 5. Generate Answer

The retrieved context is provided to Gemini.

### 6. Display Answer

The application displays:

```text
AI Answer
+
Source PDF
+
Page Number
```

---

## 📝 Quiz Workflow

```text
Selected PDFs
      ↓
Relevant Content
      ↓
Gemini
      ↓
10 MCQs
      ↓
Student Answers
      ↓
Answer Evaluation
      ↓
Score + Explanation
      ↓
Quiz History
```

---

## 🗂️ Flashcard Workflow

```text
Study Material
      ↓
Relevant Concepts
      ↓
Gemini
      ↓
Flashcards
      ↓
Revision
```

---

## 📊 Study Dashboard

The dashboard provides an overview of the student's study activity.

It tracks:

- Questions asked
- Quiz attempts
- Quiz questions answered
- Correct answers
- Quiz accuracy
- Flashcards generated
- Flashcards reviewed
- Summaries generated

---

## 🔐 Security

Sensitive information and generated study data are excluded from the repository.

```text
.env
venv/
chroma_db/
documents/
documents.json
quiz_history.json
study_stats.json
```

The Gemini API key should never be committed to GitHub.

---

## 🚀 Future Improvements

- User authentication
- Cloud database
- Per-user quiz history
- Cloud document storage
- Weak-topic detection
- Personalized study recommendations
- Voice-based questions
- Advanced RAG retrieval
- Public cloud deployment
- Mobile application

---

## 👨‍💻 Author

**Ansh Shah**

MCA Student  
MIT-WPU

GitHub:  
https://github.com/Anshshah1145

---

## ⭐ Project

If you find StudyMate AI useful, consider giving the repository a ⭐.
