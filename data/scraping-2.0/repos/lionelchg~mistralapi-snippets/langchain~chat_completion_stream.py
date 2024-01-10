import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

if __name__ == "__main__":
    chat_cfg = {
        "api_key": os.environ["MISTRAL_API_KEY"],
        "base_url": "https://api.mistral.ai/v1/",
        "model": "mistral-tiny"
    }
    chat = ChatOpenAI(**chat_cfg)
    for chunk in chat.stream("Explain in two sentences what is a mixture of experts."):
        print(chunk.content, end="", flush=True)