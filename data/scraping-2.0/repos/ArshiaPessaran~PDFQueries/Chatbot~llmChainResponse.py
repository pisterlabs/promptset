import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.agents import load_tools, initialize_agent, AgentType
from pdfDocLoader import vectordb
# from llamaapi import LlamaAPI
# llama = LlamaAPI("Your_API_Token")
# from langchain_experimental.llms import ChatLlamaAPI

# OpenAI API Key
from APIKey import OpenAIKey
if OpenAIKey is None:
    raise ValueError("OPEN_API_KEY is not set. Create APIKey.py file that defines variable OpenAIKey")
os.environ["OPENAI_API_KEY"] = OpenAIKey

# Chatbot Chain
chatBotResponse = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo"),
    vectordb.as_retriever(),
    return_source_documents=False,
    verbose=False
)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}------------------------------------")
print('Make queries about your documents')
print('------------------------------------')
while True:
    query = input(f"{green}Prompt (type q to quit): ")
    if query == "q":
        sys.exit()
    if query == '':
        continue
    result = chatBotResponse(
        {"question": query,
         "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))