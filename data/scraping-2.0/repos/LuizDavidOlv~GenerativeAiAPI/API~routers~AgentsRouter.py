import os
from fastapi import APIRouter
from langchain.agents import tool, AgentExecutor, initialize_agent, AgentType, load_tools
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import Tool
from langchain.tools.render import format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.chains import LLMMathChain
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage

router = APIRouter(
    prefix="/agents",
    tags=["Agents"]
)

SerpAPIWrapper.serpapi_api_key = os.environ.get('SERPAPI_API_KEY')
chat_history = []

class CalculatorInput(BaseModel):
    question: str = Field()

@router.post("/word-length/")
def word_length(text: str):
    toolsConfig= [get_word_length]
    llmConfig = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_with_tools = llmConfig.bind(functions=[format_tool_to_openai_function(t) for t in toolsConfig])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are very powerful assistant, but bad at calculating lengths of words",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agentConfig = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent = agentConfig, tools = toolsConfig, verbose = True)
    result = agent_executor.invoke({"input": text})
    return result["output"]

@router.post("/current-events/")
def current_events(question: str):
    llmConfig = ChatOpenAI( temperature=0)
    search = SerpAPIWrapper()
    tools = [
        Tool.from_function(
            func = search.run,
            name = 'Search',
            description = "useful for when you need to answer questions about current envents",
        )
    ]

    agent = initialize_agent(tools, llmConfig, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)
    return agent.run(question)

@router.post("/current-events-modifying-existing-tools/")
def current_events(question: str = "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"):
    llmConfig = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = load_tools(["serpapi", "llm-math"],llm = llmConfig)
    tools[0].name = "Google Search"
    agent = initialize_agent(tools, llmConfig, agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)
    return agent.run(question)

@router.post("/agent-with-memory/")
def prompt_with_memory( question: str):
    toolsConfig= [get_word_length]
    llmConfig = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_with_tools = llmConfig.bind(functions=[format_tool_to_openai_function(t) for t in toolsConfig])
    MEMORY_KEY = "chat_history"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Your are a very powerfull assistant, but bad at calculating lengths of words",
            ),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agentConfig = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agentConfig, tools=toolsConfig, verbose=True)
    preResult1 = agent_executor.invoke({"input": question, "chat_history": chat_history})

    chat_history.extend(
        [
            HumanMessage(content= question),
            AIMessage(content = preResult1["output"]),
        ]
    )

    sponsoredMessage = "Always try to sell something based on the conversation"
    preResult2 = agent_executor.invoke({"input": sponsoredMessage, "chat_history": chat_history})
    
    chat_history.extend(
        [
            HumanMessage(content = sponsoredMessage),
            AIMessage(content = preResult2["output"]),
        ]
    )

    result = {
        "conversation": chat_history,
    }

    return {
        "conversation": chat_history
    }




@tool  
def get_word_length(word: str) -> int:
    """Returns the length of a word"""
    return len(word)



