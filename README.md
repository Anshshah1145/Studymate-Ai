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
📁 Project Structure
Studymate-Ai/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── src/

The following files/folders are intentionally not included in GitHub:
.env
venv/
chroma_db/
documents/
quiz_history.json
study_stats.json

⚙️ Installation
1. Clone the repository
git clone https://github.com/Anshshah1145/Studymate-Ai.git

2. Open the project
cd Studymate-Ai

3. Create a virtual environment
python3 -m venv venv

4. Activate the virtual environment
macOS / Linux
source venv/bin/activate

Windows
venv\Scripts\activate

5. Install dependencies
pip install -r requirements.txt

🔑 API Key Setup
StudyMate AI uses the Gemini API.
Create a file named:
.env

Add:
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY

Replace YOUR_GEMINI_API_KEY with your own Gemini API key.
Never upload .env to GitHub.
▶️ Run the Application
After installing the dependencies:
streamlit run app.py

The application will open in your browser.
🔄 How It Works
1. Upload PDF
The user uploads one or more study PDFs.
2. Process Documents
StudyMate AI:
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB

3. Ask a Question
The user asks a question about their study material.
4. Retrieve Relevant Information
ChromaDB performs similarity search and retrieves the most relevant document chunks.
5. Generate Answer
The retrieved context is sent to Gemini.
6. Display Answer
The application displays:
AI Answer
+
Source PDF
+
Page Number

📝 Quiz Workflow
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

🗂️ Flashcard Workflow
Study Material
      ↓
Relevant Concepts
      ↓
Gemini
      ↓
Flashcards
      ↓
Revision

🔐 Security
Sensitive and generated files are excluded from the repository.
.env
venv/
chroma_db/
documents/
quiz_history.json
study_stats.json

The Gemini API key should never be committed to GitHub.
🚀 Future Improvements
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
👨‍💻 Author
Ansh Shah
MCA Student
MIT-WPU
GitHub:
https://github.com/Anshshah1145
⭐ Project
If you find StudyMate AI useful, consider giving the repository a ⭐.

### Put it on GitHub

From your Terminal:

```bash
cd ~/rag-studymate
nano README.md
