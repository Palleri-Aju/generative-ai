from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an experienced teacher. Explain concepts clearly for {audience}."
    ),
    (
        "human",
        "Explain {topic} at a {difficulty} level in {lines} lines."
    )
])

chain = prompt | model

audience = input("Audience: ")
topic = input("Topic: ")
difficulty = input("Difficulty level: ")
lines = input("Number of lines: ")

response = chain.invoke({
    "audience": audience,
    "topic": topic,
    "difficulty": difficulty,
    "lines": lines
})

print("\nResponse:")
print(response.content[0]["text"])