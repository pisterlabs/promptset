import os

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


os.environ['OPENAI_API_KEY'] = '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'

llm = OpenAI()
chat_model = ChatOpenAI()

text = "can you introduce AI?"

if __name__ == "__main__":
    print(llm.invoke(text))
    print(chat_model.invoke(text))