import openai
import os
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from prompts import llm_prompt, vlm_prompt, get_agent_prompt

default_llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
tools = load_tools(["wikipedia"], llm=default_llm)

def llm_agent_response(query, chat_history, llm=default_llm):
    react_agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=False)
    try :
        prompt = get_agent_prompt(query, chat_history)
        response = react_agent.run(prompt)
        return response
    except Exception as e:
        print("exception from agent")
        response = str(e)
        if response.startswith("Could not parse LLM output: `"):
            response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
            return response
        return "The agent doesn't have valid answer. Therefore you should look for a valid answer"


def vlm_response(query, img, chat_history, agent_response, report_question="", llm=default_llm):
    template = vlm_prompt
    prompt = ChatPromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    return chain.run({'agent_output':agent_response, 'input':query, 'llm_history':chat_history, 'report_question':report_question})

def llm_response(query, chat_history, report_question="", llm=default_llm):
    #agent_response = llm_agent_response(query, chat_history)
    template = llm_prompt
    prompt = ChatPromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    response = chain.run({'input':query, 'llm_history':chat_history, 'report_question':report_question})
    print("llm response ", response)
    return response



