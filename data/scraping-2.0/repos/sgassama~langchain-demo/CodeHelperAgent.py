from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper, TextRequestsWrapper, PythonREPL, StackExchangeAPIWrapper
from langchain.tools import YouTubeSearchTool
from langchain.chat_models import ChatOpenAI
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI


import chainlit as cl

@cl.on_chat_start
def start():
    search = GoogleSearchAPIWrapper()
    # requests = TextRequestsWrapper()
    # python_repl = PythonREPL()
    # stackexchange = StackExchangeAPIWrapper()
    # yt_search = YouTubeSearchTool()

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for when you need to search things up in general or need to look up software or coding references/examples",
        ),
        # Tool(
        #     name="Requests",
        #     func=requests.get,
        #     description="Useful for when you need to make http requests to external URLs",
        # ),
        # Tool(
        #     name="python_repl",
        #     description="Usefule for when you need to run pytonn code or repl.",
        #     func=python_repl.run,
        # ),
        # Tool(
        #     name="Stack Exchange",
        #     description="Usefule for when you want to search technical-based information. You can also use this when finding information about software development.",
        #     func=stackexchange.run,
        # ),
        # Tool(
        #     name="Youtube search",
        #     description="Usefule for when you want to search youtube.",
        #     func=yt_search.run,
        # ),
    ]

    prefix = """You are an enthusiastic computer software and coding expert that helps users solve problems and answer questions. 
    THINK STEP BY STEP! Use the tone of a coding tutor and provide thorough explanations.
    Respond with written/generated code examples/snippets along with explanations of each snippet. 
    Also provide references and APIs for your responses if necessary.
    Be enthusiatic in your responses. DON'T FORGET TO PROVIDE CODE EXAMPLES!!!
    You have access to the following tools: [Search, Stack Exchange]"""
    
    suffix = """Begin!"
    {chat_history}
    
    User Input: {input}
    
    {agent_scratchpad}
    """

    FORMAT_INSTRUCTIONS = """
    To use a tool, please use the following format:
    '''
    Thought: Do I need to use a tool? Yes
    Action: the action to take, one of [Search, Stack Exchange]
    Action Input: the input to the action
    Observation: the result of the action
    '''
    This cycle of tool usage can happen/repeat as many times as needed until an answer is found.

    When you have gathered enough information, write it out to the user using the following format:
    '''
    Final Answer: [respond to the user]
    '''

    If you do not know the answer, answer with "Final Answer: I don't know". 
    """

    # llm = ChatGoogleGenerativeAI(streaming=True, model="gemini-pro", temperature=0.2, max_retries=5)
    llm  = ChatOpenAI(streaming=True, model="gpt-3.5-turbo", temperature=1, max_retries=5)
    # llm  = ChatOpenAI(streaming=True, model="gpt-4", temperature=0.5, max_retries=3)

    # 
    prompt = ZeroShotAgent.create_prompt(
        tools,
        input_variables=["input", "chat_history", "agent_scratchpad"],
        prefix=prefix,
        format_instructions=FORMAT_INSTRUCTIONS,
        suffix=suffix
    )
    # 
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # 
    llm_with_stop = llm.bind(stop=["\nObservation"])
    # 
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_stop
        | ReActSingleInputOutputParser()
    )
    # 
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True
    )
    # 
    cl.user_session.set("agent_chain", agent_chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("agent_chain")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    await chain.ainvoke(
        {"input": message.content},
        config=RunnableConfig(callbacks=[cb]),
    )