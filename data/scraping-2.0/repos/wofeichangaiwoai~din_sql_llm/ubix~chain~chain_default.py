from langchain import LLMChain, SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

from ubix.common.llm import llm


from langchain.chains.router import MultiPromptChain

from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

def get_default_chain(llm):

    prompt_template = "{question}"
    prompt = PromptTemplate(
        input_variables=["question"], template=prompt_template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


if __name__ == '__main__':
    default_chain = get_default_chain(llm)
