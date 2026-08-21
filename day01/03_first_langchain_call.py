from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

client = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite");

# LangChain's Gemini integration checks GOOGLE_API_KEY first and uses GEMINI_API_KEY as a fallback.
# You do not need to specify the base_url, as LangChain's Gemini integration uses the default base_url for Gemini API.    

response = client.invoke("Explain the cocept of Galaxy in 3 lines.");

# print(response)

print(response.content[0]["text"])

