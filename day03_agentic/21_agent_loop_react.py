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
# 3. MODEL
# ============================================================

# Using 3.7 because you experienced temporary high demand on 3.8.
MODEL = "gemini-3.7-flash"


# ============================================================
# 4. ACTUAL PYTHON TOOL 1
# ============================================================

def get_product_price(product):

    prices = {
        "laptop": 50000,
        "phone": 30000,
        "tablet": 20000
    }

    return prices.get(
        product.lower(),
        "Product not found"
    )


# ============================================================
# 5. ACTUAL PYTHON TOOL 2
# ============================================================

def calculate_final_price(price, discount_percent):

    discount = price * discount_percent / 100

    final_price = price - discount

    return {
        "original_price": price,
        "discount": discount,
        "final_price": final_price
    }


# ============================================================
# 6. TOOL DECLARATION 1
# ============================================================

get_product_price_declaration = {

    "type": "function",

    "name": "get_product_price",

    "description": (
        "Gets the current price of a product. "
        "Use this whenever the user asks for the price "
        "of a laptop, phone or tablet."
    ),

    "parameters": {

        "type": "object",

        "properties": {

            "product": {
                "type": "string",
                "description": "Name of the product"
            }
        },

        "required": [
            "product"
        ]
    }
}


# ============================================================
# 7. TOOL DECLARATION 2
# ============================================================

calculate_final_price_declaration = {

    "type": "function",

    "name": "calculate_final_price",

    "description": (
        "Calculates the final price after applying "
        "a percentage discount to the original price."
    ),

    "parameters": {

        "type": "object",

        "properties": {

            "price": {
                "type": "number",
                "description": "Original product price"
            },

            "discount_percent": {
                "type": "number",
                "description": "Discount percentage"
            }
        },

        "required": [
            "price",
            "discount_percent"
        ]
    }
}


# ============================================================
# 8. ALL TOOLS
# ============================================================

tools = [

    get_product_price_declaration,

    calculate_final_price_declaration
]


# ============================================================
# 9. TOOL REGISTRY
# ============================================================

tool_registry = {

    "get_product_price": get_product_price,

    "calculate_final_price": calculate_final_price
}


# ============================================================
# 10. USER QUESTION
# ============================================================

user_input = input("Ask me something: ")


# ============================================================
# 11. INITIAL STATE
# ============================================================

previous_interaction_id = None

iteration = 0

MAX_ITERATIONS = 10


# ============================================================
# 12. AGENT LOOP
# ============================================================

while True:

    iteration += 1

    print("\n================================")
    print(f"AGENT ITERATION {iteration}")
    print("================================")


    # --------------------------------------------------------
    # Safety: prevent infinite loop
    # --------------------------------------------------------

    if iteration > MAX_ITERATIONS:
        print("Maximum agent iterations reached.")
        break


    # --------------------------------------------------------
    # Ask Gemini what to do next
    # --------------------------------------------------------

    interaction = client.interactions.create(

        model=MODEL,

        input=user_input,

        tools=tools,

        previous_interaction_id=previous_interaction_id
    )


    # --------------------------------------------------------
    # Collect function results
    # --------------------------------------------------------

    function_results = []


    # --------------------------------------------------------
    # Examine steps returned by Gemini
    # --------------------------------------------------------

    for step in interaction.steps:

        if step.type == "function_call":

            print("\nLLM selected tool:")
            print(step.name)

            print("\nArguments:")
            print(step.arguments)


            # ------------------------------------------------
            # Find actual Python function
            # ------------------------------------------------

            tool_function = tool_registry.get(
                step.name
            )


            if tool_function is None:

                raise ValueError(
                    f"Unknown tool: {step.name}"
                )


            # ------------------------------------------------
            # Convert arguments to dictionary
            # ------------------------------------------------

            arguments = dict(
                step.arguments
            )


            # ------------------------------------------------
            # Execute function dynamically
            # ------------------------------------------------

            result = tool_function(
                **arguments
            )


            print("\nTool result:")
            print(result)


            # ------------------------------------------------
            # Build result to send back to Gemini
            # ------------------------------------------------

            function_results.append(

                {
                    "type": "function_result",

                    "name": step.name,

                    "call_id": step.id,

                    "result": [

                        {
                            "type": "text",

                            "text": json.dumps(
                                result
                            )
                        }
                    ]
                }
            )


    # ========================================================
    # 13. IF NO FUNCTION CALL → AGENT IS FINISHED
    # ========================================================

    if not function_results:

        print("\n================================")
        print("FINAL ANSWER")
        print("================================")

        print(interaction.output_text)

        break


    # ========================================================
    # 14. SEND OBSERVATIONS BACK TO GEMINI
    # ========================================================

    user_input = function_results

    previous_interaction_id = interaction.id