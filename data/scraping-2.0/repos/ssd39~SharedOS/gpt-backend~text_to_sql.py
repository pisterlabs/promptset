from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.agent import AgentExecutor
import json


def getSqlQuery(opena_ai_key, db, question):
    llm = OpenAI(openai_api_key=opena_ai_key, temperature=0, verbose=True)

    verbose = True

    SQL_PREFIX = """You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query ot run. Limit results to top 5 for observation 
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    If the question does not seem related to the database, just return "I don't know" as the answer.
    """

    SQL_SUFFIX = """Begin!

    Question: {input}
    Thought: I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables.
    {agent_scratchpad}"""

    FORMAT_INSTRUCTIONS = """Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: SQL query (this query will be not of top 5 result but orignal answer query).
    """

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    prefix = SQL_PREFIX.format(dialect=toolkit.dialect)

    tools = toolkit.get_tools()

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=SQL_SUFFIX,
        format_instructions=FORMAT_INSTRUCTIONS,
        input_variables=None,
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=None,
    )
    

    tool_names = [tool.name for tool in tools]

    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)

    agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            callback_manager=None,
            verbose=verbose,
            max_iterations=15,
            max_execution_time=None,
            early_stopping_method="force",
        )

    return agent_executor.run(question)

def getVegaLiteSpec(opena_ai_key, db, query, question, sampleRow):
    llm = OpenAI(openai_api_key=opena_ai_key,temperature=0, verbose=True, )

    verbose = True

    SQL_PREFIX = """You are an agent designed to give Vega-Lite specification for given data.
    Given a question, sql query for that question and 1 result row of query. You need to decide appropriate graph to visualize data perfectly.
    """

    SQL_SUFFIX = """Begin!

    Question: Question To generate sql: {question} | Sql Query:{sqlquery} | Result Row: {resultrow} 
    Thought: I should look at question, sql query and sql result.  Then I should decide which graph will be appropriate.
    """

    resultrow = sampleRow
    
    FORMAT_INSTRUCTIONS = """Use the following format:
    Vega-Lite specification (in JSON formt):
    """
    prompt = ZeroShotAgent.create_prompt(
        tools=[],
        prefix=SQL_PREFIX,
        suffix=SQL_SUFFIX,
        format_instructions=FORMAT_INSTRUCTIONS,
        input_variables=["question", "sqlquery", "resultrow"],
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=None,
    )
    generations = llm_chain.generate([{"question": question, "sqlquery": query, "resultrow": resultrow}]).generations
    if len(generations) > 0:
        ans = generations[0][0].text
        ans = ans.replace('Vega-Lite Specification (in JSON format):', '').strip()
        ans = ans.replace('\n', '')
        return json.loads(ans)
    return {}

'''
question = "Top 10 brands with highest number of products"
query = getSqlQuery(db, question)
spec = getVegaLiteSpec(db, query, question)
print(query, spec)
'''