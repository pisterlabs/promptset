from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from utils.tools import QueryCheckerTool, InfoSQLDatabaseTool, ListSQLDatabaseTool, FormatBigNumbers, QuerySQLDatabaseTool
from utils.prompts import SQL_PREFIX, SQL_SUFFIX, QUERY_CHECKER, FORMAT_INSTRUCTIONS

def get_sql_agent_executor(llm, db, top_k = 10, dialect = 'sqlite', verbose = True, max_iterations = 10):
    tools = [
        QuerySQLDatabaseTool(db=db),
        InfoSQLDatabaseTool(db=db),
        ListSQLDatabaseTool(db=db),
        QueryCheckerTool(db=db),
    ]

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix = SQL_PREFIX.format(top_k=top_k, dialect=dialect),
        suffix=SQL_SUFFIX,
        input_variables=["input", "agent_scratchpad", "chat_history"],
    )
    llm_chain = LLMChain(llm = llm, prompt = prompt)
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain, allowed_tools=tool_names)
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5)
    agent_executor = AgentExecutor(agent, tools=tools, memory=memory, max_iterations=max_iterations, verbose=verbose)
    return agent_executor
