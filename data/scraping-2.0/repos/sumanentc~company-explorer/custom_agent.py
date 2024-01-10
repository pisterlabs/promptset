import re
from datetime import datetime
from typing import Optional, List, Union

import requests
from langchain import LLMMathChain, LLMChain
from langchain.agents import AgentOutputParser, LLMSingleActionAgent, AgentExecutor, initialize_agent, AgentType
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.chat_models import ChatOpenAI
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import HumanMessage, AgentAction, AgentFinish
from langchain.tools import BaseTool, DuckDuckGoSearchRun, Tool

# langchain.debug = True
from llm_agent import get_income_statement, get_balance_sheet, get_cash_flow, get_earnings
from prompt_helper import get_generic_answer_prompt


def search_general(input_text):
    return DuckDuckGoSearchRun().run(f"{input_text}")


def search_income_statement(input_text):
    return DuckDuckGoSearchRun().run(f"site:finance.yahoo.com income statement of {input_text}")


def search_balance_sheet(input_text):
    return DuckDuckGoSearchRun().run(f"site:finance.yahoo.com balance sheet of {input_text}")


def search_cash_flow(input_text):
    return DuckDuckGoSearchRun().run(f"site:finance.yahoo.com cash flow of {input_text}")


def search_earnings(input_text):
    return DuckDuckGoSearchRun().run(f"site:finance.yahoo.com Earnings of {input_text}")


def search_income_statement(input_text):
    return DuckDuckGoSearchRun().run(f"site:finance.yahoo.com income statement of {input_text}")


def search_balance_sheet(input_text):
    return DuckDuckGoSearchRun().run(f"site:finance.yahoo.com balance sheet of {input_text}")


def search_cash_flow(input_text):
    return DuckDuckGoSearchRun().run(f"site:finance.yahoo.com cash flow of {input_text}")


def search_earnings(input_text):
    return DuckDuckGoSearchRun().run(f"site:finance.yahoo.com Earnings of {input_text}")


def get_annual_income_statement(input_text):
    data = get_income_statement()
    annual_documents = []
    if data:
        for annual_rep in data.get('annualReports'):
            annual_documents.extend(
                f'{key} : {value}'
                for key, value in annual_rep.items()
                if value and value != '' and value != 'None'
            )
        if annual_documents:
            annual_resp = ', '.join(annual_documents)
            return annual_resp
        else:
            return search_income_statement(input_text)
    else:
        return search_income_statement(input_text)


def get_quarterly_income_statement(input_text):
    data = get_income_statement()
    quarterly_documents = []
    currentYear = datetime.now().year
    if data:
        for quater_rep in data.get('quarterlyReports'):
            if str(currentYear) in quater_rep.get('fiscalDateEnding') or str(currentYear - 1) in quater_rep.get(
                    'fiscalDateEnding'):
                quarterly_documents.extend(
                    f'{key} : {value}'
                    for key, value in quater_rep.items()
                    if value and value != '' and value != 'None'
                )
        if quarterly_documents:
            quat_resp = ', '.join(quarterly_documents)
            return quat_resp
        else:
            return search_income_statement(input_text)
    else:
        return search_income_statement(input_text)


def get_annual_balance_sheet(input_text):
    data = get_balance_sheet()
    annual_documents = []
    if data:
        for annual_rep in data.get('annualReports'):
            annual_documents.extend(
                f'{key} : {value}'
                for key, value in annual_rep.items()
                if value and value != '' and value != 'None'
            )
        if annual_documents:
            annual_resp = ', '.join(annual_documents)
            return annual_resp
        else:
            return search_balance_sheet(input_text)
    else:
        return search_balance_sheet(input_text)


def get_quarterly_balance_sheet(input_text):
    data = get_balance_sheet()
    quarterly_documents = []
    currentYear = datetime.now().year
    if data:
        for quater_rep in data.get('quarterlyReports'):
            if str(currentYear) in quater_rep.get('fiscalDateEnding') or str(currentYear - 1) in quater_rep.get(
                    'fiscalDateEnding'):
                quarterly_documents.extend(
                    f'{key} : {value}'
                    for key, value in quater_rep.items()
                    if value and value != '' and value != 'None'
                )
        if quarterly_documents:
            quat_resp = ', '.join(quarterly_documents)
            return quat_resp
        else:
            return search_balance_sheet(input_text)
    else:
        return search_balance_sheet(input_text)


def get_annual_cash_flow(query):
    data = get_cash_flow()
    annual_documents = []
    if data:
        # open_api_key = st.session_state.openai_api_key
        for annual_rep in data.get('annualReports'):
            annual_documents.extend(
                f'{key} : {value}'
                for key, value in annual_rep.items()
                if value and value != '' and value != 'None'
            )
        if annual_documents:
            annual_resp = ', '.join(annual_documents)
            return annual_resp
        else:
            return search_cash_flow(query)
    else:
        return search_cash_flow(query)


def get_quarterly_cash_flow(query):
    data = get_cash_flow()
    quarterly_documents = []
    currentYear = datetime.now().year
    if data:
        # open_api_key = st.session_state.openai_api_key
        for quater_rep in data.get('quarterlyReports'):
            if str(currentYear) in quater_rep.get('fiscalDateEnding') or str(currentYear - 1) in quater_rep.get(
                    'fiscalDateEnding'):
                quarterly_documents.extend(
                    f'{key} : {value}'
                    for key, value in quater_rep.items()
                    if value and value != '' and value != 'None'
                )
        if quarterly_documents:
            quat_resp = ', '.join(quater_rep)
            return quat_resp
        else:
            return search_cash_flow(query)
    else:
        return search_cash_flow(query)


def get_annual_earnings(query):
    data = get_earnings()
    annual_documents = []
    if data:
        # open_api_key = st.session_state.openai_api_key
        for annual_rep in data.get('annualEarnings'):
            annual_documents.extend(
                f'{key} : {value}'
                for key, value in annual_rep.items()
                if value and value != '' and value != 'None'
            )
        if annual_documents:
            annual_resp = ', '.join(annual_documents)
            return annual_resp
        else:
            return search_earnings(query)
    else:
        return search_earnings(query)


def get_quarterly_earnings(query):
    data = get_earnings()
    quarterly_documents = []
    currentYear = datetime.now().year
    if data:
        # open_api_key = st.session_state.openai_api_key
        for quater_rep in data.get('quarterlyEarnings'):
            if str(currentYear) in quater_rep.get('fiscalDateEnding') or str(currentYear - 1) in quater_rep.get(
                    'fiscalDateEnding'):
                quarterly_documents.extend(
                    f'{key} : {value}'
                    for key, value in quater_rep.items()
                    if value and value != '' and value != 'None'
                )
        if quarterly_documents:
            quat_resp = ', '.join(quater_rep)
            return quat_resp
        else:
            return search_earnings(query)
    else:
        return search_earnings(query)


def get_company_overview(query):
    import streamlit as st
    comp_ticker = st.session_state.selected_ticker
    print(f'\n selected tick {comp_ticker}')
    data = None
    name = st.session_state.company_name
    if comp_ticker:
        api_key = st.secrets["alpha_vantage_api_key"]
        url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={comp_ticker}&apikey={api_key}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        r = requests.get(url, headers=headers)
        data = r.json()
    res_list = []
    if data:
        for key, value in data.items():
            if value and value != '' and value != 'None':
                res_list.append(key + ' : ' + value)
        if not res_list:
            res_list.append('name: ' + name)
        result = ', '.join(res_list)
        if res_list:
            return result
        else:
            return search_general(query)
    else:
        return search_general(query)


class CompanyOverviewSearchTool(BaseTool):
    name = "Company Overview"
    description = "useful for when you need to answer questions about company overview or basic information about the company"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        import streamlit as st
        comp_ticker = st.session_state.selected_ticker
        print(f'\n selected ticker {comp_ticker}')
        data = None
        name = st.session_state.company_name
        if comp_ticker:
            api_key = st.secrets["alpha_vantage_api_key"]
            url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={comp_ticker}&apikey={api_key}'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
            r = requests.get(url, headers=headers)
            data = r.json()
        res_list = []
        if data:
            for key, value in data.items():
                if value and value != '' and value != 'None':
                    res_list.append(key + ' : ' + value)
            if not res_list:
                res_list.append('name: ' + name)
            result = ', '.join(res_list)
            if res_list:
                return result
            else:
                return search_general(query)
        else:
            return search_general(query)

    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Company Overview does not support async")


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)

        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts

        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        # regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        # If it can't parse the output it raises an error
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match[1].strip()
        action_input = match[2]

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def get_llm_response(input: str, open_api_key: str):
    llm = ChatOpenAI(temperature=0.1, model_name='gpt-3.5-turbo-0613', openai_api_key=open_api_key, max_retries=2, )
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)
    tools = [
        Tool(
            name="Company Annual Income statements",
            func=get_annual_income_statement,
            description="useful to answer questions about company's annual income, revenue growth," \
                        " financial performance over a specific period, assessing profitability, growth, and performance",
        ),
        Tool(
            name="Company Quarterly Income statements",
            func=get_quarterly_income_statement,
            description="useful to answer questions about company's quarterly or quarter on quarter income, revenue growth," \
                        " financial performance over a specific period, assessing profitability, growth, and performance"
        ),
        Tool(
            name="Company Annual Balance Sheet",
            func=get_annual_balance_sheet,
            description="useful to answer questions about company's annual prospects," \
                        " financial stability, financial position, creditworthiness, balance sheet"
        ),
        Tool(
            name="Company Quarterly Balance Sheet",
            func=get_quarterly_balance_sheet,
            description="useful to answer questions about company's quarterly or quarter on quarter prospects," \
                        " financial stability, financial position, creditworthiness,  balance sheet"
        ),
        Tool(
            name="Company Annual Cash Flow",
            func=get_annual_cash_flow,
            description="useful for when you need to answer questions about company's annual cash generation, liquidity, and financial health, cash position"
        ),
        Tool(
            name="Company Quarterly Cash Flow",
            func=get_quarterly_cash_flow,
            description="useful for when you need to answer questions about company's quarterly cash generation, liquidity, and financial health, cash position"
        ),
        Tool(
            name="Company Annual Earnings",
            func=get_annual_earnings,
            description="useful for when you need to answer questions about company's yearly earnings reports provide a broader, long-term perspective on a company's financial health, strategic direction, and governance."
        ),
        Tool(
            name="Company Overview",
            func=get_company_overview,
            description="useful for when you need to answer questions about company overview or basic information about the company"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to calculate something"
        ),
    ]

    # https://github.com/hwchase17/langchain/issues/1358

    # Set up the prompt with input variables for tools, user input and a scratchpad for the model to record its workings
    template = """You role is to respond to Business user questions based on the context.
                  Answer the following questions as an expert Business Analyst to the best of your ability.
                  You have access to the following tools:
    
    {tools}
    
    Use the following format:
    - Question: the input question you must answer
    - Thought: Do I need to use a tool? Yes
    - Action: the action to take, should be one of [{tool_names}]
    - Action Input: Provide the input required for the chosen tool, 
    - Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    
    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the following format(the prefix of "Thought: " and "Final Answer: " are must be included):
    
    Thought: Do I need to use a tool? No
    Final Answer: [your response here]

    Begin! Remember to speak as a passionate and an expert business analyst when giving your final answer.
    
    Question: {input}
    {agent_scratchpad}"""

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    output_parser = CustomOutputParser()

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Using tools, the LLM chain and output_parser to make an agent
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        # We use "Observation" as our stop sequence so it will stop when it receives Tool output
        # If you change your prompt template you'll need to adjust this as well
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    # Initiate the agent that will respond to our queries
    # Set verbose=True to share the CoT reasoning the LLM goes through
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=10,
                                                        handle_parsing_errors="Check your output and make sure it conforms!")
    try:
        agent_response = agent_executor.run(input)
    except ValueError as e:
        print(f'Value Error: {e}')
        response = str(e)
        agent_response = (
            response.removeprefix("Could not parse LLM output: `").removesuffix("`")
            if response.startswith("Could not parse LLM output: `")
            else "Sorry, Couldn't help you at this moment. Please try sometime later"
        )
    except Exception as e:
        print(e)
        agent_response = "Sorry, Couldn't help you at this moment. Please try sometime later"
    print("Got the result from OpenAI")
    return agent_response


def get_final_answer(openapi_key, query):
    import streamlit as st
    matched_token = st.session_state.matched_token
    if matched_token:
        import streamlit as st
        name = st.session_state.company_name
        query = re.sub(matched_token, name, query, flags=re.IGNORECASE)
    print(query)
    try:
        return get_llm_response(get_generic_answer_prompt(query), openapi_key)
    except Exception as e:
        print(f'Exception occurred {e}')
        return "Sorry, couldn't process the request now. Try again after sometime"


def get_company_income(user_query, open_api_key):
    print(user_query)
    tools = [Tool(name="Company Income", func=search_income_statement,
                  description="useful to answer questions about company's income, revenue growth, "
                              "financial performance , assessing profitability, growth, "
                              "or performance over a specific period"
                              "you are not able to understand", ),
             Tool(name="Search the internet", func=search_general,
                  description="useful for when you need to search some information that is missing or "
                              "you are not able to understand", ), ]
    llm = ChatOpenAI(temperature=0.1, model_name='gpt-3.5-turbo-0613', openai_api_key=open_api_key, max_retries=2, )
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)
    return agent.run(user_query)
