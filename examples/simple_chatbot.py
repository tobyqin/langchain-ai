import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Get the API key, base URL, and model name from environment variables
api_key = os.getenv("API_KEY")
api_base = os.getenv("API_BASE")
model_name = os.getenv("MODEL_NAME")

# Check if the required environment variables are set
if not all([api_key, api_base, model_name]):
    raise ValueError(
        "Please make sure you have set the API_KEY, API_BASE, and MODEL_NAME environment variables in your .env file."
    )

# Initialize the ChatOpenAI model
llm = ChatOpenAI(
    model_name=model_name,
    openai_api_key=api_key,
    openai_api_base=api_base,
)

print("Simple Chatbot is ready. Type 'exit' to quit.")

# Main conversation loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Create a list of messages for the conversation
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=user_input),
    ]

    # Get the response from the LLM
    try:
        response = llm.invoke(messages)
        print(f"Bot: {response.content}")
    except Exception as e:
        print(f"An error occurred: {e}")
