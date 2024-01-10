import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from decouple import config
from langchain.chains import LLMChain
import coloredlogs, logging
import streamlit as st
from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', default='INFO'))

CHAT_MODEL = 'gpt-3.5-turbo-16k'

def build_charts(question: str, df: pd.DataFrame, open_ai_model: str) -> None:
    df.to_csv('data.csv', index=False)
    csv = 'data.csv'
    pd_agent = create_csv_agent(
        ChatOpenAI(temperature=0, model=open_ai_model),
        csv,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )
    # prompt = f"""
    #     Answer this question about the data:
    #     {question}

    #     Don't call the action "`python_repl_ast`" (this won't work) - instead just use "python_repl_ast".
    #     """
    # result = pd_agent.run(prompt)
    # return result

    prompt = f"""
        Write me a python script that uses streamlit answer a question about this data.
        Only return this python script, nothing else - no commentary / formatting.
        Assume your whole output is going to be inserted right into a python script and run.
        Run the lines of code yourself to ensure they actually work.
        Don't call the action "`python_repl_ast`" (this won't work) - instead just use "python_repl_ast".

        Answer these specific questions with your resulting python script:
        {question}
        Display results with nice charts (using streamlit or plotly.express) - if applicable.
        
        The filepath to the csv you're analyzing is ./{csv}

        Only return this python script, nothing else - no commentary / formatting.
        Assume your whole output is going to be inserted right into a python script and run.
        """

    @st.cache_data
    def ask_the_csv_agent(prompt):
        result = pd_agent.run(prompt)
        return result

    placeholder = st.empty() # we do this so 
    with placeholder:
        result = (
            ask_the_csv_agent(prompt)
            .replace('The Python script for the Streamlit dashboard is as follows:', '')
            .replace('```', '\n').replace('python', '')
        )
        result_lines = result.split('\n')
        if 'python' in result_lines[0].lower():
            result = '\n'.join(result_lines[1:])
    placeholder.empty()

    return result