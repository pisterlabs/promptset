import pandas as pd
import toml
from langchain.agents import create_csv_agent, AgentExecutor
from utils.data_tools import load_data
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI


"""
Natural Language Processing (NLP):
Implement an NLP module capable of understanding and processing user queries in human language.
The module should extract key information from the queries (e.g., "How many students enrolled for Biology this year?").

We will need to build out the following:
- A method to read the data and dynamically return the meta-data needed for an initial prompt
- A method to call a code interpreter and return the results
- A method to connect the code interpreter and the data sources

The kind of language model implementation that is able to understand, query, and process user queries into data 
science/analysis task are called Code Interpreters. There are a lot of very simple ones like Langchain's CSV Agent
and many complex ones, like the one that powers GPT-4's CI app.

This is a fairly simple analytics task (by human standards) so we'll be able to get away with a simple code interpreter.

Things that would complicate this task:
    1. generalizing the application
        Because we know the the use-case and the data, we can get very far with a simple code interpreter and a data 
        informed* prompt.
        
    2. Reducing the data sources 
        With just two csvs, we have the simplest scenario (two very common, easily standardized, and static data 
        sources) So we can probably get away with just using a LangChain CSV agent. But if we had to do this with an 
        huge number of files (eg massive datalake, or a database), a more 
        complex/verbose data type that would run us out of tokens (eg HTML, JSON, XML), a dynamic data source (eg
        a web scraper), or a complex/less popular analysis package (eg asyncio, reticulate) we would need to build a 
        more complex code interpreter. 
    
    3. The low complexity of the analysis task
        This partly comes with this being an interview question, but the examples given for the types of analysis are:
        - "How many students enrolled for Biology this year?"
        These types of queries could be answered with one simple SQL query
            `SELECT COUNT(*) FROM students WHERE major = 'Biology' AND year = '2023'`
        Getting a language model to be able to understand the data and execute this query is a very simple task. Of 
        course, this is a robust system that can handle much more than that. I only say this to color the initial ask, 
        not to imply that this system is only capable of that.
    
\* In this context, data informed means with specific, hard-coded knowledge of the data we're working with.

Note: There does not seem to be a "subject" column where we can split by Biology.


Challanges:

1. data was messy, easiest as a square and What is readable for a human isnt always readable for a computer
2. making the model context aware is hard, we need to make sure that there is a layer that sits on top of the model
    that can understand the context of the question and the data. This is the code interpreter's wrapper.
"""


def get_prompt(name, section):
    toml_path = f"data/prompts.toml"
    with open(toml_path, "r") as f:
        prompts = toml.load(f)
    return prompts[section][name]


def get_data_agent(data: dict[str, pd.DataFrame], type='csv', model='gpt-4'):
    chat_model = ChatOpenAI(model_name=model, temperature=0)
    if type == 'csv':
        agent = create_csv_agent(
            chat_model,
            [key for key in data.keys()],
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
    elif type == 'pandas':
        agent = create_pandas_dataframe_agent(
            chat_model,
            [val for val in data.values()],
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            number_of_head_rows=5
        )

    else:
        raise NotImplementedError(f"Data type {type} not implemented. Try 'csv' or 'pandas'")

    return agent


class SchoolLLM:
    model_name: str
    data_sources: dict
    system_prompt: str
    human_prompt: str
    agent: AgentExecutor = None

    def __init__(self, agent_type='csv'):
        self.data_sources = load_data()

        llm = ChatOpenAI(temperature=0, model="gpt-4")
        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
        self.data_chain = get_data_agent(self.data_sources, type=agent_type, model='gpt-4')
        tools = [
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer questions about math"
            ),
            Tool(
                name="data_analysis_tool",
                func=self.data_chain.run,
                description="Useful for when you need to answer questions about FooBar. Input should be clear and detailed, in the form of a question, and containing full context. "
            )
        ]

        self.agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
        self.agent.agent.prompt.messages[0].content = ("You are a helpful AI assistant who is an expert in Data "
                                                       "Science. Always remember to "
                                                       "group by, filter and aggregate your data to meet the needs of "
                                                       "the question. For example, if df2 has a column called "
                                                       "County, you may have to group by County to get the answer to "
                                                       "a question about counties as a whole.")

    def run(self, text, use_wrapper: bool = True, **kwargs):
        """
        Creates the agent ready to be returned to streamlit or whatever
        :return:
        """
        text = "Use the tools you were given to answer the question: " + text
        if use_wrapper:
            result = self.agent.run(text, **kwargs)
        else:
            result = self.data_chain.run(text, **kwargs)
        return result
