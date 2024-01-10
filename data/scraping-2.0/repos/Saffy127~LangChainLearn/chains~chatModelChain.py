from langchain.chatmodels import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.chat import (
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
)

human_message_prompt = HumanMessagePromptTemplate(
  prompt=PromptTemplate(
    template="What is a good name for a company that makes {product}?",
    input_variables=["product"],
  )
)

chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
chat = ChatOpenAI(temperature=0.9)
chain = LLMChain(llm=chat, prompt=chat_prompt_template)
print(chain.run("mini computers"))





