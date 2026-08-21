from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv();

client = OpenAI(
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url = os.getenv("GEMINI_BASE_URL")
);

response = client.chat.completions.create(
    model="gemini-3.5-flash-lite",
    messages = [
        {
            "role":"system",
            "content":"You are a NASA scientist that explains concepts in scientific terms."
        },
        {
            "role":"user",
            "content":"Explain the concept of Galaxy in 3 lines."
        }
    ]
)

print(response.choices[0].message.content);