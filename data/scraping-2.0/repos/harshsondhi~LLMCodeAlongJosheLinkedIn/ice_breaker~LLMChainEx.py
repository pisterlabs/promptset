from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"
api_key = "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"
llm = OpenAI()
chat = ChatOpenAI(openai_api_key=api_key, temperature=0)
embedding_function = OpenAIEmbeddings()


human_message_prompt = HumanMessagePromptTemplate.from_template(
    "Make up a funny company name for a company that produces {product}"
)
chat_prompt_templae = ChatPromptTemplate.from_messages([human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt_templae)

print(chain.run(product="Computers"))
