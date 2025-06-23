import os
from collections import deque

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Get the API key, base URL, and model name from environment variables
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
MODEL_NAME = os.getenv("MODEL_NAME")
# Set the number of recent messages to remember.
# The history includes both user and bot messages.
# A value of 5 means 5 messages will be kept.
CONVERSATION_HISTORY_LENGTH = 5

# --- Initialization ---
# Check if the required environment variables are set
if not all([API_KEY, API_BASE, MODEL_NAME]):
    raise ValueError(
        "Please make sure you have set the API_KEY, API_BASE, and MODEL_NAME environment variables in your .env file."
    )

# Initialize the ChatOpenAI model
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    openai_api_key=API_KEY,
    openai_api_base=API_BASE,
)

# Use a deque to store the conversation history with a fixed length
history = deque(maxlen=CONVERSATION_HISTORY_LENGTH)

# --- Main Loop ---
print("Contextual Chatbot is ready. Type 'exit' to quit.")
print(f"I will remember the last {CONVERSATION_HISTORY_LENGTH} messages.")

# Add a system message to set the bot's persona
system_message = SystemMessage(content="You are a helpful assistant.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    human_message = HumanMessage(content=user_input)

    # Construct the list of messages to send to the LLM
    # This includes the system message, the recent history, and the new user message
    messages_to_send = [system_message] + list(history) + [human_message]

    try:
        # Get the response from the LLM
        response = llm.invoke(messages_to_send)

        # The response itself is an AIMessage object
        bot_message = response
        print(f"Bot: {bot_message.content}")

        # Add the user's message and the bot's response to the history
        history.append(human_message)
        history.append(bot_message)

    except Exception as e:
        print(f"An error occurred: {e}")
