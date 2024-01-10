from langchain.agents import LLMSingleActionAgent
from langchain.callbacks import get_openai_callback

from agents.general_agent import GeneralAgent
from tools.general_tools import QuerySQLDataBaseTool, SQlCommentGenerator, DummyTool, MetabaseCreator
from utils.config_loaders import get_config, get_secrets
from utils.database import create_db_session

# noinspection DuplicatedCode
config = get_config()
secrets = get_secrets(config)

engine = create_db_session(secrets['database']['url'])
agent = GeneralAgent(engine=engine, verbose=True)


class TestGeneralInstance:

    def test_define_tools(self):
        tools = agent.define_tools()
        assert len(tools) == 4
        assert isinstance(tools[0], DummyTool)
        assert isinstance(tools[1], QuerySQLDataBaseTool)
        assert isinstance(tools[2], SQlCommentGenerator)
        assert isinstance(tools[3], MetabaseCreator)

    def test_get_tools(self):
        tools = agent.get_tools("Give me a dummy tool")
        assert len(tools) > 0
        assert isinstance(tools[0], DummyTool)

    def test_define_agent(self):
        agent_obj = agent.define_agent(agent.TEMPLATE, agent.define_tools())
        assert isinstance(agent_obj, LLMSingleActionAgent)

    def test_handle_request(self):
        with get_openai_callback() as cb:
            response = agent.handle_request("How many rows are in the store table from sales schema?")
            assert "Error" not in response
            assert cb.successful_requests > 0


# TODO: Commented out because it is too expensive to run
# class TestInstance:
#
#     def test_general_agent_demo(self):
#         with get_openai_callback() as cb:
#             result = agent.handle_request("How many rows are in the store table from sales schema?")
#             print(result)
#             print(f"Total Tokens: {cb.total_tokens}")
#             print(f"Prompt Tokens: {cb.prompt_tokens}")
#             print(f"Completion Tokens: {cb.completion_tokens}")
#             print(f"Successful Requests: {cb.successful_requests}")
#             print(f"Total Cost (USD): ${cb.total_cost}")
#
#     def test_metabase_request(self):
#         with get_openai_callback() as cb:
#             result = agent.handle_request("""
# I want a dashboard that shows me the total profit per month, number of sales per month, sales by status, number of sales by shipping date, sales by territory, and sales by type of credit card.
# You can use the tables salesorderheader, salesorderdetail, specialofferproduct, store, creditcard, customer, and salesterritory from the sales schema.
#             """)
#             print(result)
#             print(f"Total Tokens: {cb.total_tokens}")
#             print(f"Prompt Tokens: {cb.prompt_tokens}")
#             print(f"Completion Tokens: {cb.completion_tokens}")
#             print(f"Successful Requests: {cb.successful_requests}")
#             print(f"Total Cost (USD): ${cb.total_cost}")
