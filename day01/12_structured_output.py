from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# Define the structure we expect from the LLM
class TopicExplanation(BaseModel):
    topic: str = Field(
        description="The topic being explained"
    )

    definition: str = Field(
        description="A simple definition of the topic"
    )

    key_concepts: list[str] = Field(
        description="Important concepts related to the topic"
    )

    difficulty: str = Field(
        description="Difficulty level of the explanation"
    )


model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# Ask LangChain to return output matching our Pydantic model
structured_model = model.with_structured_output(
    TopicExplanation
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an experienced technology teacher."
    ),
    (
        "human",
        """
        Explain {topic} for a {difficulty} learner.

        Identify the important concepts the learner should understand.
        """
    )
])


chain = prompt | structured_model


topic = input("Topic: ")
difficulty = input("Difficulty: ")


response = chain.invoke({
    "topic": topic,
    "difficulty": difficulty
})


print(response)