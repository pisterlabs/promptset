import json
from datetime import datetime

import requests
import streamlit as st
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import DuckDuckGoSearchRun
from langchain_community.tools.render import format_tool_to_openai_function
from langchain_core.prompts import MessagesPlaceholder

from function import CompanyIncomeStatementTool, SearchTool, CompanyBalanceSheetTool, CompanyCashFlowTool, \
    CompanyInformationTool, CompanyEarningsTool, SearchCompanyNewsTool
from prompt_helper import get_company_name_prompt, get_income_statement_prompt, get_balance_sheet_prompt, \
    get_cash_flow_prompt, get_earnings_prompt


def run(prompt: str, api_key: str, json_result=True, ):
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0613', openai_api_key=api_key, max_retries=2,
                     max_tokens=500)
    messages = [
        SystemMessage(
            content="You are an expert business analyst."
        ),
        HumanMessage(
            content=prompt
        ),
    ]
    try:
        response = llm(messages)
    except Exception as e:
        print(e)
        return json.loads("")
    return json.loads(response.content) if json_result else response.content


def run_with_search(prompt, api_key):
    messages = [
        SystemMessage(
            content="You are an expert business analyst."
        ),
        HumanMessage(
            content=prompt
        ),
    ]
    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613", openai_api_key=api_key, streaming=True,
                         max_retries=2, max_tokens=1000)
        search = DuckDuckGoSearchRun(name="Search")
        search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                        handle_parsing_errors=True)
        response = search_agent.run(messages)
        return response
    except Exception as e:
        print(f'Exception occurred {e}')
        return "Sorry, couldn't process the request now. Try again after sometime"


def get_company_name(user_query, api_key):
    company_name_prompt = get_company_name_prompt(user_query)
    return run(company_name_prompt, api_key)


def get_ticker(company_name, api_key):
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={api_key}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    r = requests.get(url, headers=headers)
    data = r.json()
    print(data)
    distinct_comp_name = set()
    ticker_comp_name_dict = {}
    if matches := data.get('bestMatches'):
        for match in matches:
            if (match["4. region"] in ["United States", "India/Bombay"]
                    and match['2. name'].replace('.', '').lower() not in distinct_comp_name):
                distinct_comp_name.add(match['2. name'].replace('.', '').lower())
                ticker_comp_name_dict[match['1. symbol']] = match['2. name']
        return ticker_comp_name_dict
    if info := data.get('Information'):
        if 'Our standard API rate limit' in info:
            st.warning(
                'Free access to Company Explorer allows for 25 daily requests. Kindly attempt again on the following day.')
            st.stop()
    return None


def get_income_statement(ticker):
    print(f'Inside get_income_statement {ticker}')
    import streamlit as st
    if ticker:
        comp_ticker = ticker
    else:
        comp_ticker = st.session_state.selected_ticker
    print(f'{comp_ticker}')
    if comp_ticker:
        api_key = st.secrets["alpha_vantage_api_key"]
        url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={comp_ticker}&apikey={api_key}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        r = requests.get(url, headers=headers)
        return r.json()
    return None


def get_balance_sheet(ticker):
    import streamlit as st
    if ticker:
        comp_ticker = ticker
    else:
        comp_ticker = st.session_state.selected_ticker
    print(f'{comp_ticker}')
    if comp_ticker:
        api_key = st.secrets["alpha_vantage_api_key"]
        url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={comp_ticker}&apikey={api_key}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        r = requests.get(url, headers=headers)
        return r.json()
    return None


def get_cash_flow(ticker):
    import streamlit as st
    if ticker:
        comp_ticker = ticker
    else:
        comp_ticker = st.session_state.selected_ticker
    print(f'{comp_ticker}')
    if comp_ticker:
        api_key = st.secrets["alpha_vantage_api_key"]
        url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={comp_ticker}&apikey={api_key}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        r = requests.get(url, headers=headers)
        return r.json()
    return None


def get_earnings(ticker):
    import streamlit as st
    if ticker:
        comp_ticker = ticker
    else:
        comp_ticker = st.session_state.selected_ticker
    print(f'{comp_ticker}')
    if comp_ticker:
        api_key = st.secrets["alpha_vantage_api_key"]
        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={comp_ticker}&apikey={api_key}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        r = requests.get(url, headers=headers)
        return r.json()
    return None


def is_income_statement_query(user_query, api_key):
    income_statement_prompt = get_income_statement_prompt(user_query)
    return run(income_statement_prompt, api_key)


def is_balance_sheet_query(user_query, api_key):
    balance_sheet_prompt = get_balance_sheet_prompt(user_query)
    return run(balance_sheet_prompt, api_key)


def is_cash_flow_query(user_query, api_key):
    cash_flow_prompt = get_cash_flow_prompt(user_query)
    return run(cash_flow_prompt, api_key)


def is_earnings_query(user_query, api_key):
    earnings_prompt = get_earnings_prompt(user_query)
    return run(earnings_prompt, api_key)


def get_data_as_string(data):
    temp_data = []
    for map in data:
        for key, value in map.items():
            data_str = f'{key} : {value}'
            temp_data.append(data_str)

    return ', '.join(temp_data)


def execute_user_query(query, api_key):
    prompt = ChatPromptTemplate.from_messages([(
        "system",
        "Answer the user questions as an expert Business Analyst to the best of your ability."
        " \n If year is present in user question then use that year else use current year {current_year}."
    ), "{chat_history}",
        ("human", "{question}"),
        ("human",
         """Tip: 1. Remember to speak as a passionate and an expert business analyst when giving your final answer."
                 2. Provide the response backed by numbers wherever possible
                 3. Provide the output in raw markdown javascript format.
                 3. Don't mention as a passionate and expert business analyst in the response."""),
        MessagesPlaceholder(variable_name="agent_scratchpad"), ])

    tools = [CompanyIncomeStatementTool(),
             CompanyBalanceSheetTool(),
             CompanyCashFlowTool(),
             CompanyEarningsTool(),
             SearchTool(),
             SearchCompanyNewsTool()]
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, input_key='question', k=3)
    llm_with_tools = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0613', openai_api_key=api_key, max_retries=2,
                                max_tokens=500).bind(functions=[format_tool_to_openai_function(t) for t in tools])
    agent = (
            {
                "question": lambda x: x["question"],
                "current_year": lambda x: x["current_year"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3, memory=memory)
    return agent_executor.invoke(
        {"question": query, "current_year": datetime.now().year}
    )


def get_company_information(query, open_api_key):
    prompt = ChatPromptTemplate.from_messages([(
        "system",
        """Answer the user questions as an expert Business Analyst to the best of your ability.\n
         Provide a brief summary of the company overview
         \n If year is present in user question then use that year else use current year {current_year}."""
    ),
        ("human", "{question}"),
        ("human",
         """Tip: 1. Remember to speak as a passionate and an expert business analyst when giving your final answer."
                 2. Provide the output in raw markdown javascript format.
                 3. Don't mention as a passionate and expert business analyst in the response."""),
        MessagesPlaceholder(variable_name="agent_scratchpad"), ])

    tools = [CompanyInformationTool(),
             SearchTool()]
    llm_with_tools = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0613', openai_api_key=open_api_key,
                                max_retries=2,
                                max_tokens=200).bind(functions=[format_tool_to_openai_function(t) for t in tools])
    agent = (
            {
                "question": lambda x: x["question"],
                "current_year": lambda x: x["current_year"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
    )
    print(f'Query {query}')
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor.invoke(
        {"question": query, "current_year": datetime.now().year}
    )
