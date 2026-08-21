from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv;

load_dotenv();

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite");

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Python assistant that explains concepts in simple terms."),
    ("human", "Explain {topic} at a {difficulty} level in {lines} lines.")
])

messages = prompt_template.invoke({
    "audience": "software developers",
    "topic": "Generative AI",
    "difficulty": "beginner",
    "lines": 5
})

response = model.invoke(messages)

print(response.content[0]["text"])
