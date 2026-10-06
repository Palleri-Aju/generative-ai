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
# 3. ACTUAL PYTHON TOOL 1 - CALCULATOR
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
# 4. ACTUAL PYTHON TOOL 2 - TEXT ANALYZER
# ============================================================

def text_analyzer(text, operation):

    if operation == "word_count":
        return len(text.split())

    elif operation == "character_count":
        return len(text)

    elif operation == "uppercase":
        return text.upper()

    else:
        return "Invalid operation"


# ============================================================
# 5. TOOL DECLARATION - CALCULATOR
# ============================================================

calculator_declaration = {

    "type": "function",

    "name": "calculator",

    "description": (
        "Use this tool for mathematical calculations "
        "such as addition, subtraction, multiplication and division."
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
# 6. TOOL DECLARATION - TEXT ANALYZER
# ============================================================

text_analyzer_declaration = {

    "type": "function",

    "name": "text_analyzer",

    "description": (
        "Use this tool to analyze text. "
        "It can count words, count characters or convert text to uppercase."
    ),

    "parameters": {

        "type": "object",

        "properties": {

            "text": {
                "type": "string",
                "description": "The text to analyze"
            },

            "operation": {

                "type": "string",

                "enum": [
                    "word_count",
                    "character_count",
                    "uppercase"
                ]
            }
        },

        "required": [
            "text",
            "operation"
        ]
    }
}


# ============================================================
# 7. LIST OF TOOLS AVAILABLE TO THE LLM
# ============================================================

tools = [
    calculator_declaration,
    text_analyzer_declaration
]


# ============================================================
# 8. TOOL REGISTRY
# ============================================================

tool_registry = {

    "calculator": calculator,

    "text_analyzer": text_analyzer
}


# ============================================================
# 9. GET USER QUESTION
# ============================================================

question = input("Ask me something: ")


# ============================================================
# 10. SEND USER REQUEST TO GEMINI
# ============================================================

interaction = client.interactions.create(

    model="gemini-3.7-flash",

    input=question,

    tools=tools
)


# ============================================================
# 11. FIND FUNCTION CALL
# ============================================================

function_call = next(

    (
        step

        for step in interaction.steps

        if step.type == "function_call"
    ),

    None
)


# ============================================================
# 12. CHECK WHETHER TOOL WAS SELECTED
# ============================================================

if function_call:

    print("\n--------------------------------")
    print("LLM DECISION")
    print("--------------------------------")

    print("Tool selected:")
    print(function_call.name)

    print("\nArguments:")
    print(function_call.arguments)


    # ========================================================
    # 13. GET ACTUAL PYTHON FUNCTION FROM REGISTRY
    # ========================================================

    tool_function = tool_registry.get(
        function_call.name
    )


    if tool_function is None:

        raise ValueError(
            f"Unknown tool: {function_call.name}"
        )


    # ========================================================
    # 14. GET ARGUMENTS
    # ========================================================

    arguments = dict(
        function_call.arguments
    )


    # ========================================================
    # 15. EXECUTE SELECTED TOOL
    # ========================================================

    result = tool_function(
        **arguments
    )


    print("\n--------------------------------")
    print("TOOL EXECUTION")
    print("--------------------------------")

    print("Result:")
    print(result)


    # ========================================================
    # 16. SEND TOOL RESULT BACK TO GEMINI
    # ========================================================

    final_interaction = client.interactions.create(

        model="gemini-3.7-flash",

        previous_interaction_id=interaction.id,

        tools=tools,

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


    # ========================================================
    # 17. FINAL ANSWER
    # ========================================================

    print("\n--------------------------------")
    print("FINAL LLM RESPONSE")
    print("--------------------------------")

    print(final_interaction.output_text)


# ============================================================
# 18. MODEL DECIDED NO TOOL WAS REQUIRED
# ============================================================

else:

    print("\n--------------------------------")
    print("NO TOOL REQUIRED")
    print("--------------------------------")

    print(interaction.output_text)