from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Artificial Neural Network is inspired by the human brain."

vector = embeddings.embed_query(text)

print("Vector length:", len(vector))
print("First 10 values:")
print(vector[:10])