from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()

client = genai.Client(
    api_key = os.getenv("GEMINI_API_KEY"),
)

#=========================================================
# Describe the tool to the LLM
#========================================================= 

calculator_declaration = types.FunctionDeclaration(
    name = "calculator",
    description = (
        "Calculator tool that can perform basic arithmetic operations ""such as addition, subtraction, multiplication, and division. "
        "It takes two numbers and an operation as input and returns the result of the calculation."
    ),
    parameters_json_schema = {
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
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The arithmetic operation to perform."
            }   
        },
        "required": ["a", "b", "operation"]
    }
)

#=========================================================
# Package the function declaration as a Gemini tool
#=========================================================

calculator_tool = types.Tool(
    function_declarations = [calculator_declaration]
)

#=========================================================
# Ask User a Question
#=========================================================

question = input("Enter a question for the LLM (e.g., 'What is 5 plus 3?'): ")

#=========================================================
# Send the question and the tool description to the LLM
#=========================================================

response = client.models.generate_content(
    model = "gemini-3.5-flash-lite",
    contents = question,
    config = types.GenerateContentConfig(
        tools = [calculator_tool],
        automatic_function_calling = types.AutomaticFunctionCallingConfig(
           disable = True,
        )
    )
)

#=========================================================
# Check if the LLM wants to call the calculator tool   
#=========================================================

if response.function_calls:
    function_call = response.function_calls[0]
    print(f"LLM wants to call the function: {function_call.name}")
    print(f"Function arguments: {function_call.args}")
else:
    print("LLM did not request a function call.")
    print(f"Response: {response.contents}")
