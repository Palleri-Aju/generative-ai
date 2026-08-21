from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url = os.getenv("GEMINI_BASE_URL") 
)

response = client.chat.completions.create(
    model="gemini-3.5-flash-lite",
    messages = [
        {
            "role": "system",
            "content": "You are a helpful Python teacher."
        },
        {
            "role": "user", 
            "content": "Explain Python lists in 3 lines."
        }
    ]);

print(response.choices[0].message.content);

print("\nToken usage details:")

# print(response.usage);

# prompt_tokens
# Everything you sent to the model.
# That includes things such as:
# System message + User message + Previous chat history + RAG context + Tool results 
print(f"Prompt tokens used: {response.usage.prompt_tokens}")

# completion_tokens
# Everything the model sent back to you.
print(f"Completion tokens used: {response.usage.completion_tokens}")

# total_tokens 
print(f"Total tokens used: {response.usage.total_tokens}")

# ┌─────────────────────────────────────┐
# │           CONTEXT WINDOW            │
# │                                     │
# │ System instructions                 │
# │                                     │
# │ Previous conversation               │
# │                                     │
# │ Current user message                │
# │                                     │
# │ RAG documents                       │
# │                                     │
# │ Tool results                        │
# │                                     │
# │ Model response                      │
# │                                     │
# └─────────────────────────────────────┘