from langchain.llms.openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


def create_agent(file):
    llm = OpenAI()
    df = pd.read_csv(file)
    # Create a Pandas DataFrame agent.
    return create_pandas_dataframe_agent(llm, df, verbose=False)


def get_response(agent, query):
    prompt_template = '''
    For the following query,
    If the query requires drawing a table, response as follows:
    {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

    If the query requires creating a bar chart, response as follows:
    {"bar": {"columns": ["A", "B", "C", ...], "data": [10, 20, ...]}}

    If the query requires creating a line chart, response as follows:
    {"line": {"columns": ["A", "B", "C", ...], "data": [10, 25, ...]}}

    There can only be two types of chart, "bar" and "line".

    If it is just asking a question that requires neither, response as follows:
    {"answer": "answer"}
    For Example:
    {"answer": "The product with the highest price is 'A-2'"}

    If you do not know the answer, don't make up anything, response as follows:
    {"answer": "I do not know. Information not in the context."}

    Return the response in a JSON format.

    All strings in the "columns" list and the "data" list, should be always in double quotes:
    For example:
    {"columns": ["product", "price"],
    "data": [["A", 320], ["B", 1000]]}

    Process step by step and verify whether the above conditions are all satisfied in the response.
    If not, reconsider the response.

    Below is the query.
    Query:
    '''
    response = agent.run(prompt_template + query)
    return response.__str__()
