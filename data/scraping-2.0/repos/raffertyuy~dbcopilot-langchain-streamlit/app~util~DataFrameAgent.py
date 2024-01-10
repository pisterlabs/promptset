from langchain.agents import create_pandas_dataframe_agent

from . import Prompts

class DataFrameAgent:
    """
    A LangChain Pandas DataFrame Agent helper class.

    Args:
        df: The dataframe.
        llm: The language model to use for the agent.
        verbose: Whether to print verbose output.

    Attributes:
        agent: The agent used for querying the dataframe.

    Methods:
        Query(query): Query the agent and return the response as a string.
    """

    def __init__(self, df, llm, verbose=False):
        self.agent = create_pandas_dataframe_agent(llm=llm, df=df, verbose=verbose)

    # Functions
    def Query(self, query):
        """
        Query an agent and return the response as a string.

        Args:
            query: The query to ask the agent.

        Returns:
            The response from the agent as a string.
        """

        prompt = Prompts.QUERY_PROMPT + query
        response = self.agent.run(prompt)

        return response.__str__()