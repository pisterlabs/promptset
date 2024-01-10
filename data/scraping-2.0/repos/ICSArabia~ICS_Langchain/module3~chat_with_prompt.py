import os
from langchain.llms import OpenAI
from flask import jsonify
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

MODEL_NAME = 'gpt-3.5-turbo-1106'
CWD = os.getcwd()
FILENAME = os.path.join(CWD, 'module3', 'letters.txt')

def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def chat_with_prompt(user_query, stylevar):
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.8)

    style_test = read_file_content(FILENAME)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if stylevar is None:
        prompt = ChatPromptTemplate.from_template(f"You are ICS Arabia chatbot so answer {user_query} accordingly. If the following question or text contains any financial data related to ICS Arabia, reply with: 'Sharing sensitive financial information is a violation of the company policy.")

        chatbot = LLMChain(llm=llm, prompt=prompt, memory=memory)
    else:
        prompt2 = ChatPromptTemplate.from_template(f"You are ICS Arabia's AI assistance: rewrite the following: {user_query} in the style of the following text: {style_test}")

        chatbot = LLMChain(llm=llm, prompt=prompt2, memory=memory)

    response = chatbot({"query": user_query})
    return response["text"]

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    style = input("Enter the style: ")
    result = chat_with_prompt(user_query, "ICS_Style")
    print(result)
