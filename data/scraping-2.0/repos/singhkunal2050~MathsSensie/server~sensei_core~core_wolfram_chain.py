import logging
import os
from prompt_helpers.prompts import get_prompt_for_solution, get_prompt_for_doubts_of_solution

from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# wolfram = WolframAlphaAPIWrapper()

# def use_wolfram(questiondef):
#     return wolfram.run(question)
memory = ConversationBufferMemory(memory_key="chat_history")

# os.environ["OPENAI_API_KEY"]
# os.environ["WOLFRAM_ALPHA_APPID"]

def initialize_wolfram_agent():
    logging.info("initializing wolfram agent")
    llm = OpenAI(temperature=0)
    logging.info("initializing wolfram tool")
    tools = load_tools(['wolfram-alpha'])
    agent =  initialize_agent(tools, llm, agent="conversational-react-description", memory=memory, verbose=True)
    return agent

def ask_wolfram_alpha(agent, question, frame_context):
    prompt = get_prompt_for_doubts_of_solution(question) if frame_context == "doubt" else get_prompt_for_solution(question)
    conversation_history = memory.load_memory_variables({})
    return agent.run(prompt)
