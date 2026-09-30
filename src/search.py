from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to existing ChromaDB
vector_store = Chroma(
    collection_name="studymate",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# 3. Ask a question
question = "What is an Artificial Neural Network?"

# 4. Search for the most relevant chunks
results = vector_store.similarity_search(
    question,
    k=3
)

# 5. Display results
print("\nQuestion:")
print(question)

print("\nRelevant chunks:\n")

for i, result in enumerate(results):
    print(f"--- Result {i + 1} ---")
    print(result.page_content)
    print("Page:", result.metadata.get("page_label"))
    print()