from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CODE_REVIEW_PROMPT = """
Review this piece of code and offer potential refactoring options. 

CODE - 
 
{text}

"""

UNIT_TEST_MAKING_PROMPT = """
Write unit tests for this code! 

CODE - 

{text}

"""
DEBUG_PROMPT = """
Check this piece of code for bugs!

CODE - 

{text}
"""


def create_code_review_chain():
    prompt = ChatPromptTemplate.from_template(CODE_REVIEW_PROMPT)
    model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    chain = prompt | model
    return chain

def create_unit_test_making_chain():
    prompt = ChatPromptTemplate.from_template(UNIT_TEST_MAKING_PROMPT)
    model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    chain = prompt | model
    return chain

def create_debug_chain():
    prompt = ChatPromptTemplate.from_template(DEBUG_PROMPT)
    model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    chain = prompt | model
    return chain


