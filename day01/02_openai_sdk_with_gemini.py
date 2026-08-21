from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key = os.getenv("GEMINI_API_KEY"),
    # base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    base_url = os.getenv("GEMINI_BASE_URL"))
    
response = client.chat.completions.create(
    model="gemini-3.5-flash-lite",
    messages = [
        {
            "role": "user", 
            "content": "Explain Mother in 3 lines."
        }
    ]
);

# print(response);

print(response.choices[0].message.content);