from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from dotenv import load_dotenv;

load_dotenv();

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite");

messages = [
    SystemMessage("You are a helpful Python assistant that explains concepts in simple terms.")
];

while True:
    user_input = input("User: ");
    if user_input.lower() == "exit":
        break;

    messages.append(HumanMessage(content = user_input));
    
    response = model.invoke(messages);
    
    assistant_response = response.content[0]["text"];
    print(f"Assistant: {assistant_response}");

    messages.append(AIMessage(content = assistant_response));
