import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()

OUTPUT_FILE = "data.jsonl"
CHAT = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

messages = [
    HumanMessage(
        content="Come up with a question as follows:"
    ),
]
CHAT(messages).content
