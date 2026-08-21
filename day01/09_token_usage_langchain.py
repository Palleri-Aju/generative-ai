from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

messages = [
    SystemMessage(
        content="You are a helpful Python teacher."
    ),
    HumanMessage(
        content="Explain Python lists in 3 lines."
    )
]

response = model.invoke(messages)

print("Response:")
print(response.content[0]["text"])

print("\nToken Usage:")
print(response.usage_metadata)