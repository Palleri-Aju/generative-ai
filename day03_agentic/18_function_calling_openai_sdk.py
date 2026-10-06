from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# =========================================================
# Describe the tool to the LLM
# =========================================================

calculator_declaration = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Calculator tool that can perform basic arithmetic operations "
            "such as addition, subtraction, multiplication, and division. "
            "It takes two numbers and an operation as input."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The first number for the calculation."
                },
                "b": {
                    "type": "number",
                    "description": "The second number for the calculation."
                },
                "operation": {
                    "type": "string",
                    "enum": [
                        "add",
                        "subtract",
                        "multiply",
                        "divide"
                    ],
                    "description": "The arithmetic operation to perform."
                }
            },
            "required": ["a", "b", "operation"]
        }
    }
}

# =========================================================
# Ask User a Question
# =========================================================

question = input(
    "Enter a question for the LLM "
    "(e.g., 'What is 5 plus 3?'): "
)

# =========================================================
# Send the question and the tool description to the LLM
# =========================================================

response = client.chat.completions.create(
    model="gemini-3.5-flash-lite",

    messages=[
        {
            "role": "user",
            "content": question
        }
    ],

    tools=[
        calculator_declaration
    ],

    tool_choice={
        "type": "function",
        "function": {
            "name": "calculator"
        }
    }
)

# =========================================================
# Check if the LLM wants to call the calculator tool
# =========================================================

message = response.choices[0].message

if message.tool_calls:

    function_call = message.tool_calls[0]

    print(
        f"LLM wants to call the function: "
        f"{function_call.function.name}"
    )

    arguments = json.loads(
        function_call.function.arguments
    )

    print(
        f"Function arguments: {arguments}"
    )

else:

    print("LLM did not request a function call.")
    print(f"Response: {message.content}")