from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string
)
from openai_func import ChatGPT
from history import history_selector

"""
organizer.py
Core File. Organizer creates valid Prompt.

This Document Mainly Implements:
    1. 
    2. 
"""


def messages_tiktoken(ChatGPT:llm, messages):
    llm = llm
    return llm.calculate_tokens(get_buffer_string(messages))

def add_message(current_messages, message):
    return current_messages.append(HumanMessage(content=message))

def add_QA(current_messages, Q, A):
    current_messages.append(HumanMessage(content=Q))
    current_messages.append(AIMessage(content=A))
    return current_messages

def create_messages()

def history_organizer(llm, user_input, history_list, vec_history_list, tokens_limit=1000):
    llm = llm
    history_QAlist = []
    history_Qlist = []
    history_Alist = []
    history_timestamplist = []
    tokens = 0
    tokens = tokens + llm.calculate_tokens(str(user_input))
    tokens_list = []
    count = 0
    for history_item in history_list:
        if(tokens>tokens_limit):
            break
        history_QA = history_item[0]
        history_Q = history_item[0][0]
        history_A = history_item[0][1]
        history_QAlist.append(history_item[0])
        history_Qlist.append(history_item[0][0])
        history_Alist.append(history_item[0][1])
        history_timestamplist.append(history_item[1])
        t = llm.calculate_tokens(str(history_A)) + llm.calculate_tokens(str(history_Q))
        tokens_list.append(t)
        tokens = tokens + t
        count = count+1
    return_list = [count, tokens, tokens_list, history_QAlist, history_Qlist, history_Alist]
    
    messages =  

    return return_list


messages2 = [
        SystemMessage(content="You are an assistant."),
        HumanMessage(content="Hello, my name is Larry Zhang."),
        AIMessage(content="Hi Larry Zhang, How can I help you."),
        HumanMessage(content="k")
    ]







def main_organizer(llm, user_input):
    messages = ''
    #input = input_preprocessing(user_input)
    input = user_input


    return messages