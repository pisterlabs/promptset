from traceloop.sdk import Traceloop
import os
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from traceloop.sdk.decorators import workflow, task

Traceloop.init(app_name="openai-obs", disable_batch=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

@task(name="add_prompt_context")
def add_prompt_context():
    llm = OpenAI(openai_api_key=openai.api_key)
    prompt = ChatPromptTemplate.from_template("explain the business of company {company} in a max of {length} words")
    model = ChatOpenAI()
    chain = prompt | model
    return chain

@task(name="prep_prompt_chain")
def prep_prompt_chain():
    return add_prompt_context()

@workflow(name="ask_question")
def prompt_question():
    chain = prep_prompt_chain()
    return chain.invoke({"company": "dynatrace", "length" : 50})

if  __name__ == "__main__":
    print(prompt_question())