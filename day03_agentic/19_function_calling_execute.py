import os
import json

from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. LOAD API KEY
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")


# ============================================================
# 2. CREATE CLIENT
# ============================================================

client = genai.Client(
    api_key=api_key
)


# ============================================================
# 3. ACTUAL PYTHON FUNCTION
# ============================================================

def calculator(a, b, operation):

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":

        if b == 0:
            return "Cannot divide by zero"

        return a / b

    else:
        return "Invalid operation"


# ============================================================
# 4. TOOL DECLARATION
# ============================================================

calculator_declaration = {

    "type": "function",

    "name": "calculator",

    "description": (
        "Performs basic arithmetic operations such as "
        "addition, subtraction, multiplication and division."
    ),

    "parameters": {

        "type": "object",

        "properties": {

            "a": {
                "type": "number",
                "description": "First number"
            },

            "b": {
                "type": "number",
                "description": "Second number"
            },

            "operation": {

                "type": "string",

                "enum": [
                    "add",
                    "subtract",
                    "multiply",
                    "divide"
                ]
            }
        },

        "required": [
            "a",
            "b",
            "operation"
        ]
    }
}


# ============================================================
# 5. TAKE USER INPUT
# ============================================================

question = input("Ask a calculation question: ")


# ============================================================
# 6. SEND QUESTION + TOOL TO GEMINI
# ============================================================

interaction = client.interactions.create(

    model="gemini-3.7-flash",

    input=question,

    tools=[
        calculator_declaration
    ]
)


# ============================================================
# 7. FIND FUNCTION CALL
# ============================================================

function_call = None

for step in interaction.steps:

    if step.type == "function_call":
        function_call = step
        break

# function_call = next(
#     (
#         step
#         for step in interaction.steps
#         if step.type == "function_call"
#     ),
#     None
# )


# ============================================================
# 8. EXECUTE FUNCTION
# ============================================================

if function_call:

    print("\nLLM selected tool:")
    print(function_call.name)

    print("\nLLM generated arguments:")
    print(function_call.arguments)

    args = function_call.arguments


    if function_call.name == "calculator":

        result = calculator(
            a=args["a"],
            b=args["b"],
            operation=args["operation"]
        )

        print("\nTool result:")
        print(result)


        # ====================================================
        # 9. SEND TOOL RESULT BACK TO GEMINI
        # ====================================================

        final_interaction = client.interactions.create(

            model="gemini-3.8-flash",

            previous_interaction_id=interaction.id,

            tools=[
                calculator_declaration
            ],

            input=[
                {
                    "type": "function_result",

                    "name": function_call.name,

                    "call_id": function_call.id,

                    "result": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "result": result
                                }
                            )
                        }
                    ]
                }
            ]
        )


        # ====================================================
        # 10. FINAL RESPONSE
        # ====================================================

        print("\nFinal LLM response:")
        print(final_interaction.output_text)


else:

    print("\nNo tool required.")
    print(interaction.output_text)