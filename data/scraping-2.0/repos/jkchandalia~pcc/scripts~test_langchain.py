from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI


#import os
from dotenv import load_dotenv

load_dotenv()


from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

#print(prompt.format(product="podcast player"))
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

chatopenai = ChatOpenAI(
                model_name="gpt-3.5-turbo")
llmchain_chat = LLMChain(llm=chatopenai, prompt=prompt)
print(llmchain_chat.run("podcast player"))

'''
memory = ConversationBufferMemory()
import os
llm = ChatOpenAI()
tools = load_tools([
    'wikipedia',
    #'google-search',
    #'google-translate',
    #'termial',
    #'news-api',
    #'podcast-api',
    #'openweather-api',
    #'arxiv-api',
    #'python-repl',
    ], llm=llm)

agent= initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)
#search_me = "hello world!"
#out = agent({"input": search_me, "chat_history": []})
'''