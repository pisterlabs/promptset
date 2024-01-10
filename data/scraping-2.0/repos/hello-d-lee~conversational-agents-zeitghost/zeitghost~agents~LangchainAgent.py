#from langchain import LLMChain
from langchain.chains import LLMChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents import create_vectorstore_agent
from langchain.agents.agent_toolkits import VectorStoreInfo
from langchain.agents.agent_toolkits import VectorStoreToolkit
from langchain.agents.agent import AgentExecutor
from langchain.schema import LLMResult
from langchain.sql_database import SQLDatabase
from zeitghost.agents.Helpers import core_prompt, vector_prompt
from zeitghost.agents.Helpers import BQ_PREFIX, bq_template, core_template, vector_template
from zeitghost.agents.Helpers import base_llm, bq_agent_llm, pandas_agent_llm, vectorstore_agent_llm
from zeitghost.agents.Helpers import MyCustomHandler
from zeitghost.vertex.MatchingEngineVectorstore import MatchingEngineVectorStore
from zeitghost.vertex.LLM import VertexLLM
from langchain.callbacks.manager import CallbackManager


class LangchainAgent:
    """
    A class used to represent an llm agent to ask questions of.
    Contains agents for Pandas DataFrames, BigQuery, and Vertex Matching Engine.
    """
    callback_handler = MyCustomHandler()
    callback_manager = CallbackManager([callback_handler])

    def get_vectorstore_agent(
        self
        , vectorstore: MatchingEngineVectorStore
        , vectorstore_name: str
        , vectorstore_description: str
        , llm: VertexLLM = vectorstore_agent_llm
    ) -> AgentExecutor:
        """
        Gets a langchain agent to query against a Matching Engine vectorstore

        :param llm: zeitghost.vertex.LLM.VertexLLM
        :param vectorstore_description: str
        :param vectorstore_name: str
        :param vectorstore: zeitghost.vertex.MatchingEngineVectorstore.MatchingEngine

        :return langchain.agents.agent.AgentExecutor:
        """
        vectorstore_info = VectorStoreInfo(
            name=vectorstore_name
            , description=vectorstore_description
            , vectorstore=vectorstore
        )
        vectorstore_toolkit = VectorStoreToolkit(
            vectorstore_info=vectorstore_info
            , llm=llm
        )
        return create_vectorstore_agent(
            llm=llm
            , toolkit=vectorstore_toolkit
            , verbose=True
            , callback_manager=self.callback_manager
            , return_intermediate_steps=True
        )

    def get_pandas_agent(
        self
        , dataframe
        , llm=pandas_agent_llm
    ) -> AgentExecutor:
        """
        Gets a langchain agent to query against a pandas dataframe

        :param llm: zeitghost.vertex.llm.VertexLLM
        :param dataframe: pandas.DataFrame
            Input dataframe for agent to interact with

        :return: langchain.agents.agent.AgentExecutor
        """
        return create_pandas_dataframe_agent(
            llm=llm
            , df=dataframe
            , verbose=True
            , callback_manager=self.callback_manager
            , return_intermediate_steps=True
        )

    def get_bigquery_agent(
        self
        , project_id='cpg-cdp'
        , dataset='google_trends_my_project'
        , llm=bq_agent_llm
    ) -> AgentExecutor:
        """
        Gets a langchain agent to query against a BigQuery dataset

        :param llm: zeitghost.vertex.llm.VertexLLM
        :param dataset:
        :param project_id: str
            Google Cloud Project ID

        :return: langchain.SQLDatabaseChain
        """
        db = SQLDatabase.from_uri(f"bigquery://{project_id}/{dataset}")
        toolkit = SQLDatabaseToolkit(llm=llm, db=db)

        return create_sql_agent(
            llm=llm
            , toolkit=toolkit
            , verbose=True
            , prefix=BQ_PREFIX
            , callback_manager=self.callback_manager
            , return_intermediate_steps=True
        )

    def query_bq_agent(
        self
        , agent: AgentExecutor
        , table: str
        , prompt: str
    ) -> str:
        """
        Queries a BQ Agent given a table and a prompt.

        :param agent: AgentExecutor
        :param table: str
            Table to ask question against
        :param prompt: str
            Question prompt

        :return: Dict[str, Any]
        """

        return agent.run(
            bq_template.format(prompt=prompt, table=table)
        )

    def query_pandas_agent(
        self
        , agent: AgentExecutor
        , prompt: str
    ) -> str:
        """
        Queries a BQ Agent given a table and a prompt.

        :param agent: langchain.
        :param prompt: str
            Question prompt

        :return: Dict[str, Any]
        """

        return agent.run(
            core_template.format(question=prompt)
        )

    def query_vectorstore_agent(
        self
        , agent: AgentExecutor
        , prompt: str
        , vectorstore_name: str
    ):
        """
        Queries a VectorStore Agent given a prompt

        :param vectorstore_name:
        :param agent: AgentExecutor
        :param prompt: str

        :return: str
        """
        return agent.run(
            vector_template.format(question=prompt, name=vectorstore_name)
        )

    def chain_questions(self, questions) -> LLMResult:
        """
        Executes a chain of questions against the configured LLM
        :param questions: list(str)
            A list of questions to ask the llm

        :return: langchain.schema.LLMResult
        """
        llm_chain = LLMChain(prompt=core_prompt, llm=vectorstore_agent_llm)
        res = llm_chain.generate(questions)

        return res

