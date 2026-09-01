from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Embeddings convert text into numerical vectors that represent semantic meaning.
# Texts with similar meanings usually produce vectors that are closer together.
# These vectors are used for semantic search, similarity comparison, and RAG.
# A chat model generates text, while an embedding model generates vectors.

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

text = "The quick brown fox jumps over the lazy dog and fled the scene."

vector = embeddings_model.embed_query(text)

print("Original Text:", text)
# print("Embedding Vector:", vector)
print("Vector Length:", len(vector))
print("Vector Type:", type(vector))
print("First 5 Elements of Vector:", vector[:5])