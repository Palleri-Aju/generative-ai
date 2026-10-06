from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# STEP 1: LOAD PDF
# =========================================================

PDF_PATH = "./documents/genai_notes.pdf"

loader = PyPDFLoader(PDF_PATH)

pages = loader.load()


print("\nPages loaded:", len(pages))


# =========================================================
# STEP 2: CHECK PDF DOCUMENTS
# =========================================================

for index, page in enumerate(pages):
    
    if index >= 3: 
        break

    print(f"\n--- PAGE {index + 1} ---")

    print(page.page_content[:500])

    print("\nMetadata:")

    print(page.metadata)


# =========================================================
# STEP 3: CHUNK THE PDF
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


chunks = text_splitter.split_documents(pages)


print("\nTotal chunks created:", len(chunks))


# =========================================================
# STEP 4: DISPLAY CHUNKS
# =========================================================

for index, chunk in enumerate(chunks):
    
    if index >= 3: 
        break

    print(f"\n--- CHUNK {index + 1} ---")

    print(chunk.page_content)

    print("\nMetadata:")

    print(chunk.metadata)