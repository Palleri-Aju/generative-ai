from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv;

load_dotenv();

client = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite");

messages = [
    SystemMessage(
        content = "You are a helpful assistant that explains concepts in simple terms."
    ),
    HumanMessage(
        content = "Explain the concept of Galaxy in 3 lines"
    )
];

response = client.invoke(messages);

print(response.content[0]["text"])

