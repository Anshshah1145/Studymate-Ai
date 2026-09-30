from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# 1. Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to ChromaDB
vector_store = Chroma(
    collection_name="studymate",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# 3. Create Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

# 4. User question
question = "What is an Artificial Neural Network?"

# 5. Retrieve relevant chunks
results = vector_store.similarity_search(
    question,
    k=3
)

# 6. Combine retrieved chunks
context = "\n\n".join(
    document.page_content
    for document in results
)

# 7. Create prompt
prompt = f"""
Answer the question using ONLY the information provided in the context.

Context:
{context}

Question:
{question}

Give a clear and simple answer.
"""

# 8. Send context + question to Gemini
response = llm.invoke(prompt)

# 9. Display answer
print("\nQuestion:")
print(question)

print("\nAnswer:")
if isinstance(response.content, str):
    print(response.content)
else:
    for block in response.content:
        if block.get("type") == "text":
            print(block.get("text", ""))