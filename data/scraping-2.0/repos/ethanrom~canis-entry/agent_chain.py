from langchain.chains import LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.llms import OpenAI

from memory import memory
from tools import zeroshot_tools

import config
import pandas as pd
import os




os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
temperature = 0


def read_first_3_rows():
    dataset_path = "dataset.csv"
    try:
        df = pd.read_csv(dataset_path)
        first_3_rows = df.head(3).to_string(index=False)
    except FileNotFoundError:
        first_3_rows = "Error: Dataset file not found."

    return first_3_rows

dataset_first_3_rows = read_first_3_rows()

CUSTOM_FORMAT_INSTRUCTIONS = """Use the following format:

User Input: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""


ZEROSHOT_PREFIX = f"""
First 3 rows of the dataset:
{dataset_first_3_rows}
====
You have access to the following tools:"""


ZEROSHOT_SUFFIX = """Begin"

{chat_history}
Customer: {input}
{agent_scratchpad}"""



def get_agent_chain():

    prompt = ZeroShotAgent.create_prompt(
        zeroshot_tools,
        prefix=ZEROSHOT_PREFIX,
        suffix=ZEROSHOT_SUFFIX,
        format_instructions=CUSTOM_FORMAT_INSTRUCTIONS,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )

    zeroshot_agent_llm = OpenAI(temperature=temperature, streaming=True)
    zeroshot_llm_chain = LLMChain(llm=zeroshot_agent_llm, prompt=prompt)
    zeroshot_agent = ZeroShotAgent(llm_chain=zeroshot_llm_chain)
    zeroshot_agent_chain = AgentExecutor.from_agent_and_tools(agent=zeroshot_agent, tools=zeroshot_tools, verbose=True, memory=memory, handle_parsing_errors=True)
    return zeroshot_agent_chain