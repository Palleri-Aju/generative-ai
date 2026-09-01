from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import math

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite")

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001")

# Knowledge base sample data
knowledge_base = [
    "Python is commonly used for machine learning, data science and AI development.",
    "Java is widely used for enterprise applications and large backend systems.",
    "Mumbai is the financial capital of India and is located in Maharashtra.",
    "Tigers are large wild cats found mainly in Asia.",
    "Spider man is a fictional superhero character created by Marvel Comics.",
    "RAG combines information retrieval with a generative language model."
];

def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimensions")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)

# ---------------------------------------------------------
# STEP 1:
# Convert all documents into embedding vectors
# ---------------------------------------------------------

# document_vectors = [embeddings_model.embed_documents([doc]) for doc in knowledge_base]
document_vectors = embeddings_model.embed_documents(knowledge_base)


# ---------------------------------------------------------
# STEP 2:
# Ask the user a question
# ---------------------------------------------------------

question = input("Ask a question: ")

# ---------------------------------------------------------
# STEP 3:
# Convert the question into an embedding vector
# ---------------------------------------------------------

question_vector = embeddings_model.embed_query(question)

# ---------------------------------------------------------
# STEP 4:
# Compare question vector with every document vector
# ---------------------------------------------------------

scores = []

for document, document_vector in zip(
    knowledge_base,
    document_vectors
):
    score = cosine_similarity(
        question_vector,
        document_vector
    )

    scores.append(
        (score, document)
    )
    
# ---------------------------------------------------------
# STEP 5:
# Sort by similarity score - highest first
# ---------------------------------------------------------

scores.sort(
    key=lambda item: item[0],
    reverse=True
)


# Most relevant document
best_score, best_document = scores[0]


print("\n--- RETRIEVAL RESULTS ---")

for score, document in scores:
    print(f"\nScore: {score:.4f}")
    print("Document:", document)


print("\n--- MOST RELEVANT DOCUMENT ---")
print(best_document)

print("\nSimilarity Score:")
print(best_score)

# ---------------------------------------------------------
# STEP 6:
# Give retrieved information to the generative LLM
# ---------------------------------------------------------

messages = [

    SystemMessage(
        content="""
        You are a helpful assistant.

        Answer the user's question only using the supplied context.

        If the answer is not present in the context,
        say that the information is not available.
        """
    ),

    HumanMessage(
        content=f"""
        Context:
        {best_document}

        Question:
        {question}
        """
    )
]


# STEP 7:
# Generate final answer
# ---------------------------------------------------------

response = model.invoke(messages)


print("\n--- FINAL ANSWER ---")

if isinstance(response.content, str):
    print(response.content)
else:
    print(response.content[0]["text"])



