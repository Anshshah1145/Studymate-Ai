from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load PDF
loader = PyPDFLoader("documents/DLP UNIT 1.pdf")
documents = loader.load()

# 2. Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

# 3. Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Create ChromaDB vector store
vector_store = Chroma(
    collection_name="studymate",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# 5. Add PDF chunks to ChromaDB
vector_store.add_documents(chunks)

print("Documents added to ChromaDB successfully!")