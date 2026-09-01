from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


# =========================================================
# CONFIGURATION
# =========================================================

TOP_K = 2
SIMILARITY_THRESHOLD = 0.70


# =========================================================
# STEP 1: SOURCE KNOWLEDGE
# =========================================================

text = """
Python is a high-level programming language known for its
simple syntax and readability. It is widely used for web
development, automation, scripting and backend development.

Python is also one of the most widely used programming
languages for artificial intelligence and machine learning.
Libraries such as NumPy, Pandas, Scikit-learn, TensorFlow
and PyTorch make Python popular among AI engineers and
data scientists.

Java is a strongly typed programming language widely used
for enterprise applications. It is commonly used for large
backend systems, banking applications and Android development.

Retrieval-Augmented Generation, commonly called RAG, combines
information retrieval with generative AI. A RAG system first
retrieves relevant information from a knowledge base and then
provides that information as context to a large language model.

Embeddings represent the semantic meaning of text as numerical
vectors. Texts with similar meanings normally have vectors that
are located closer together in the embedding space.

Vector databases store embedding vectors and allow applications
to perform semantic similarity searches. Instead of searching
only for exact keywords, vector search retrieves information
based on semantic meaning.

Cosine similarity is a mathematical technique used to compare
the direction of two vectors. In semantic search, it can be
used to determine how similar a query embedding is to document
embeddings.
"""


# =========================================================
# STEP 2: CHUNK THE DOCUMENT
# =========================================================

# Large documents should not normally be embedded as one
# large piece of text.
#
# We divide the document into smaller chunks.
#
# chunk_size    -> maximum approximate characters per chunk
# chunk_overlap -> characters shared between adjacent chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)


print("\n--- GENERATED CHUNKS ---")

for index, chunk in enumerate(chunks):

    print(f"\nChunk {index + 1}:")
    print(chunk)


# =========================================================
# STEP 3: CONVERT CHUNKS INTO LANGCHAIN DOCUMENTS
# =========================================================

documents = []

for index, chunk in enumerate(chunks):

    document = Document(
        page_content=chunk,

        metadata={
            "chunk_number": index + 1
        }
    )

    documents.append(document)


# =========================================================
# STEP 4: CREATE EMBEDDING MODEL
# =========================================================

# Embeddings convert text into numerical vectors.
#
# Similar text normally produces similar vectors.
#
# These vectors are used for semantic search and RAG.

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# =========================================================
# STEP 5: CREATE VECTOR STORE
# =========================================================

vector_store = InMemoryVectorStore(
    embedding=embedding_model
)


# Add documents to the vector store.
#
# Internally:
#
# Document
#    ↓
# Embedding model
#    ↓
# Vector
#    ↓
# Vector Store

vector_store.add_documents(documents)


# =========================================================
# STEP 6: CREATE GENERATIVE MODEL
# =========================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# =========================================================
# STEP 7: CREATE RAG PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are a helpful assistant.

        Answer the user's question only using the supplied
        context.

        Do not invent information.

        If the answer is not available in the supplied context,
        say:

        "The information is not available in the provided context."
        """
    ),

    (
        "human",
        """
        Context:

        {context}


        Question:

        {question}
        """
    )

])


# =========================================================
# STEP 8: CREATE LANGCHAIN CHAIN
# =========================================================

chain = prompt | model


# =========================================================
# STEP 9: GET USER QUESTION
# =========================================================

question = input("\nAsk a question: ")


# =========================================================
# STEP 10: SEMANTIC SEARCH
# =========================================================

# TOP_K = 2
#
# This means:
#
# Retrieve at most the two most semantically similar chunks.

results = vector_store.similarity_search_with_score(
    question,
    k=TOP_K
)


# =========================================================
# STEP 11: DISPLAY RETRIEVAL RESULTS
# =========================================================

print("\n--- TOP RETRIEVAL RESULTS ---")

for document, score in results:

    print(
        f"\nChunk {document.metadata['chunk_number']}"
    )

    print(
        f"Similarity Score: {score:.4f}"
    )

    print(document.page_content)


# =========================================================
# STEP 12: APPLY SIMILARITY THRESHOLD
# =========================================================

# k controls:
#   How many top candidates we retrieve.
#
# threshold controls:
#   How relevant a candidate must be.
#
# Example:
#
# Chunk 2 -> 0.91   KEEP
# Chunk 4 -> 0.62   REJECT
#
# if threshold = 0.70


filtered_results = []

for document, score in results:

    if score >= SIMILARITY_THRESHOLD:

        filtered_results.append(
            (document, score)
        )


# =========================================================
# STEP 13: HANDLE NO RELEVANT DOCUMENTS
# =========================================================

if len(filtered_results) == 0:

    print("\n--- FINAL ANSWER ---")

    print(
        "The information is not available "
        "in the provided context."
    )

else:

    # =====================================================
    # STEP 14: DISPLAY DOCUMENTS THAT PASSED THE THRESHOLD
    # =====================================================

    print("\n--- CHUNKS SENT TO LLM ---")

    for document, score in filtered_results:

        print(
            f"\nChunk {document.metadata['chunk_number']}"
            f" | Score: {score:.4f}"
        )

        print(document.page_content)


    # =====================================================
    # STEP 15: COMBINE RETRIEVED CHUNKS INTO CONTEXT
    # =====================================================

    context = "\n\n".join(
        document.page_content
        for document, score in filtered_results
    )


    # =====================================================
    # STEP 16: SEND CONTEXT + QUESTION TO GEMINI
    # =====================================================

    response = chain.invoke({

        "context": context,

        "question": question

    })


    # =====================================================
    # STEP 17: PRINT FINAL ANSWER
    # =====================================================

    print("\n--- FINAL ANSWER ---")

    if isinstance(response.content, str):

        print(response.content)

    else:

        print(
            response.content[0]["text"]
        )