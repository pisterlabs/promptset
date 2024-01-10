import os

from langchain.agents import AgentExecutor, Tool
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.chains import APIChain, LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools.render import format_tool_to_openai_tool, render_text_description
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents.output_parsers import ReActSingleInputOutputParser

from langchain.memory import ConversationBufferMemory

with open("../opeanai_key.txt", "r") as file:
    # Read the content of the file
    OPENAI_API_KEY = file.read()

#pip install langchain, duckduckgo-searcj
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-3.5-turbo")

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

calculator = Tool(
    name="Calculator",
    func=llm_math_chain.run,
    description="useful for when you need to answer questions about math",
)


tools = [calculator ]


prompt = PromptTemplate(input_variables=['agent_scratchpad', 'chat_history', 'input', 'tool_names', 'tools'],
               template='Assistant is a coding assistant with aim to help answering coding and Machine Learning related questions'
                        '\n\nTOOLS:\n------\n\nAssistant has access to the following tools:\n\n{tools}\n\nTo use a tool,'
                        'please use the following format:\n\n```\nThought: Do I need to use a tool? Yes\nAction: the action to'
                        'take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of '
                        'the action\n```\n\nWhen you have a response to say to the Human, or if you do not need to use a tool, you MUST use'
                        'the format:\n\n```\nThought: Do I need to use a tool? No\nFinal Answer: [your response here]\n```\n\nBegin!'
                        '\n\nPreviousconversation history:\n{chat_history}'
                        '\n\nNew input: {input}\n{agent_scratchpad}')


prompt = prompt.partial(tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools]))

llm_with_stop = llm.bind(stop="\nObservation")
memory = ConversationBufferMemory(memory_key='chat_history')

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"]
    }
    | prompt
    # | llm.bind(tools=tools)
    | llm_with_stop
    | ReActSingleInputOutputParser()
)



agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

# agent_chain.invoke({"input": "What is the capital of Poland"})


# agent_chain.invoke({"input": "What was my previous question?"})
