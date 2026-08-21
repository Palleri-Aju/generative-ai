from openai import OpenAI
from dotenv import load_dotenv

import os

load_dotenv();

client = OpenAI(
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url = os.getenv("GEMINI_BASE_URL")
);

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that explains concepts in simple terms."
    }
];

while True:
    user_input = input("User: ");
    if user_input.lower() == "exit":
        break;

    messages.append({
        "role": "user",
        "content": user_input
    });

    response = client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=messages
    );

    assistant_response = response.choices[0].message.content;
    print(f"Assistant: {assistant_response}");

    messages.append({
        "role": "assistant",
        "content": assistant_response
    });

