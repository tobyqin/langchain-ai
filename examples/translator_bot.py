import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
MODEL_NAME = os.getenv("MODEL_NAME")

if not all([API_KEY, API_BASE, MODEL_NAME]):
    raise ValueError("请确保 .env 文件中配置了 API_KEY, API_BASE, MODEL_NAME")

# 预定义 prompt template
PROMPT_TEMPLATE = (
    "你是一个智能翻译助手。"
    "如果用户输入是中文或非英文，请将其翻译成英文；"
    "如果用户输入是英文，请将其翻译成中文。"
    "只输出翻译结果，不要输出其它内容。"
    "\n用户输入：{query}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_TEMPLATE),
        ("human", "{query}"),
    ]
)

llm = ChatOpenAI(
    model_name=MODEL_NAME,
    openai_api_key=API_KEY,
    openai_api_base=API_BASE,
)

print("Translator Bot is ready. Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    # 构建 prompt
    messages = prompt.format_messages(query=user_input)
    try:
        response = llm.invoke(messages)
        print(f"Bot: {response.content}")
    except Exception as e:
        print(f"An error occurred: {e}")
