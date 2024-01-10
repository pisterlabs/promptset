from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import create_csv_agent, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.output_parsers import CommaSeparatedListOutputParser

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.memory import ConversationBufferWindowMemory
import pandas as pd


load_dotenv()

item_data = os.getenv("ITEM_DATA_CSV")
item_images = os.getenv("ITEM_IMAGE_CSV")
mapped_outfits = os.getenv("MAPPED_OUTFITS_CSV")


# def pandas_agent(input=""):
#   pandas_agent_df = create_pandas_dataframe_agent(llm, df, verbose=True, openai_api_key=openai_api_key, )
#   return pandas_agent_df

# pandas_tool = Tool(
#   name='Pandas Data frame tool',
#   func=pandas_agent,
#   description="Useful for when you need to answer questions about a Pandas Dataframe"
# )

llm = OpenAI()


def csv_agent(input):
    id_agent = create_csv_agent(
        llm,
        mapped_outfits,
        verbose=True,
        # agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    return id_agent


def image_agent(input=""):
    print("ids", input)
    image_agent = create_csv_agent(
        llm,
        item_images,
        verbose=True,
    )


id_tool = Tool(
    name="ID Agent",
    func=csv_agent,
    description="Useful for finding the details of clothes specified in the input from a csv file",
)

image_tool = Tool(
    name="Image Agent",
    func=image_agent,
    description="Useful for taking the ids output by ID Agent to query the csv file for image links and returning the links.",
)

tools = [id_tool]

# conversational agent memory'prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
prefix = """"""
suffix = """

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=1, return_messages=True
)

llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

# # Create our agent
# conversational_agent = initialize_agent(
#     agent="chat-zero-shot-react-description",
#     tools=tools,
#     llm=llm,
#     verbose=True,
#     max_iterations=3,
#     early_stopping_method="generate",
#     memory=memory,
# )


# res = agent_chain.run("blue shirt (get 5 ids)")
# print(res)
res = agent_chain.run(" a top for females in the same colour (get 5 ids)")
print(res)

# agent = csv_agent("")
# # res = agent.run("Get 2 shoes where topwear = 15870 and bottomwear = 21382 or bottomwear = 23870")
# output_parser = CommaSeparatedListOutputParser()
# res = output_parser.parse(agent.run("Get 2 unique shoes where topwear = 7504 and bottomwear = 28456 or 18002 or 28458"))
# print(res, type(res))

# df = pd.read_csv(item_data)
# row = df[df["id"] == 15970]
# link = row["link"].values[0]
# print(link)

# import re

# data = ['The two unique shoes are 46086 and 36137.', '18903'] 

# pattern = r'\b\d{4,5}\b'  # Match 4 to 5 digits
# numbers = []

# for item in data:
#     matches = re.findall(pattern, item)
#     numbers.extend(matches)
#     print(matches)

# print(numbers)

# bottomwear = 28456, 18002, 28458
# footwear = 11949, 22165