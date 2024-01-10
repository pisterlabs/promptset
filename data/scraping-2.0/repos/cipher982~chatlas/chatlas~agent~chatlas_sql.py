"""Chatlas Agent for workin with SQL."""

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema.messages import AIMessage, SystemMessage
from langchain.tools.render import format_tool_to_openai_function
from langchain.utilities import SQLDatabase

from chatlas.prompts.prompts_sql import FUNCS_SUFFIX, PREFIX, SUFFIX

TOP_K = 5
INPUT_VARIABLES = None
CALLBACK_MANAGER = None
VERBOSE = True
MAX_ITERATIONS = 15
MAX_EXECUTION_TIME = None
EARLY_STOPPING_METHOD = "force"


def create_chatlas(llm: BaseChatModel, db: str, functions: bool = False) -> AgentExecutor:
    # Set db connection
    db_engine = SQLDatabase.from_uri(db)

    # Gather tools
    toolkit = SQLDatabaseToolkit(db=db_engine, llm=llm)
    tools = toolkit.get_tools()

    # Set prompts
    prefix = PREFIX.format(dialect=toolkit.dialect, top_k=TOP_K)

    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if not functions:
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=SUFFIX,
            format_instructions=FORMAT_INSTRUCTIONS,
            input_variables=INPUT_VARIABLES,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=CALLBACK_MANAGER,
        )
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    else:
        messages = [
            SystemMessage(content=prefix),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessage(content=FUNCS_SUFFIX),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        input_variables = ["input", "agent_scratchpad", "chat_history"]
        prompt = ChatPromptTemplate(input_variables=input_variables, messages=messages)

        llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

        # Define this funky runnable agent
        agent = (
            {
                "chat_history": lambda x: x["chat_history"],  # keep this first, order matters!
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_functions(x["intermediate_steps"]),
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )

    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
