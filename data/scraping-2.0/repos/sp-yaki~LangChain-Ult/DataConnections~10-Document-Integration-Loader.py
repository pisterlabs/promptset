from langchain.document_loaders import HNLoader

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

loader = HNLoader('https://news.ycombinator.com/item?id=30084169')
data = loader.load()
print(data[0].page_content)


from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
model = ChatOpenAI(temperature=0)
human_prompt = HumanMessagePromptTemplate.from_template('Please give me a single sentence summary of the following:\n{document}')
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
result = model(chat_prompt.format_prompt(document=data[0].page_content).to_messages())
print(result.content)