from langchain_community.document_loaders import PyPDFLoader

pdf_path = "documents/DLP UNIT 1.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("Number of pages:", len(documents))

print("\nFirst page content:\n")
print(documents[0].page_content)