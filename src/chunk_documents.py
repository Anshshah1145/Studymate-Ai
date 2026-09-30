from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = "documents/DLP UNIT 1.pdf"

# Load PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Create chunks
chunks = text_splitter.split_documents(documents)

print("Number of pages:", len(documents))
print("Number of chunks:", len(chunks))

# Show first chunk
print("\n--- FIRST CHUNK ---\n")
print(chunks[0].page_content)

# Show metadata
print("\n--- FIRST CHUNK METADATA ---\n")
print(chunks[0].metadata)

# Show second chunk
print("\n--- SECOND CHUNK ---\n")
print(chunks[1].page_content)