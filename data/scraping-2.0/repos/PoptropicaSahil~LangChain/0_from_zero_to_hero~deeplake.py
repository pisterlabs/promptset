import warnings


# Filter out UserWarnings - should come before the warning causing thing
warnings.filterwarnings("ignore", 
                        category=UserWarning
                        )

from dotenv import load_dotenv
import os
import logging 

from langchain.llms import OpenAI

from langchain.vectorstores import DeepLake
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# load API key from environment
openai_key = os.getenv("OPENAI_API_KEY")
activeloop_key = os.getenv("ACTIVELOOP_TOKEN")
activeloop_org_id = os.getenv("ACTIVELOOP_ORG_ID")

# # Before executing the following code, make sure to have your
# # Activeloop key saved in the “ACTIVELOOP_TOKEN” environment variable.

logging.info("Running for DeepLake Vector Store")

# # instantiate the LLM and embeddings models
llm = OpenAI(model="text-davinci-003", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# # create our documents
# texts = [
#     "Napoleon Bonaparte was born in 15 August 1769",
#     "Louis XIV was born in 5 September 1638"
# ]
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.create_documents(texts)

# logging.info(f"Docs after text_splitter are {docs}")
# # [Document(page_content='Napoleon Bonaparte was born in 15 August 1769', metadata={}), Document(page_content='Louis XIV was born in 5 September 1638', metadata={})]

# # create Deep Lake dataset
# # TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = activeloop_org_id 
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# # add documents to our Deep Lake dataset
# db.add_documents(docs)

logging.info("Done adding embeddings to DeepLake path")



# logging.info("Running for Retriving DeepLake Vector Store")


# llm = OpenAI(model="text-davinci-003", temperature=0)
# # db = DeepLake.deeplake.load('hub://sahilg/langchain_course_from_zero_to_hero')

# # create a RetrievalQA chain
# retrieval_qa = RetrievalQA.from_chain_type(
# 	llm=llm,
# 	chain_type="stuff",
# 	retriever=db.as_retriever()
# )

# tools = [
#     Tool(
#         name="Retrieval QA System",
#         func=retrieval_qa.run,
#         description="Useful for answering questions."
#     ),
# ]

# # let's create an agent that uses the RetrievalQA chain as a tool
# agent = initialize_agent(
# 	tools = tools,
# 	llm = llm,
# 	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# 	verbose=True
# )

# logging.info("Initialised the agent for Retreival QA")


# logging.info("Asking question to  Agent")

# response = agent.run("When was Napoleone born?")
# logging.info(response)


######################################################
###################################################### 
### DeepLake Vector Store - Adding more info

# load the existing Deep Lake dataset and specify the embedding function
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# create new documents
texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963",
    "Sahil played cricket and football in his school, but now plays chess"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# add documents to our Deep Lake dataset
db.add_documents(docs)
logging.info("Adding new tests to DeepLake dataset")

# recreate previous agent

# instantiate the wrapper class for GPT3
llm = OpenAI(model="text-davinci-003", temperature=0)

# create a retriever from the db
retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

# instantiate a tool that uses the retriever
tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

# create an agent that uses the tool
agent = initialize_agent(
	tools,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)

response1 = agent.run("When was Michael Jordan born?")
response2 = agent.run("Explain Sahil's choices of sports") 
# langchain.schema.OutputParserException: Parsing LLM output produced both a final answer and a parse-able action:  
# I need to find out what sports Sahil plays
# Sahil plays basketball, soccer, and tennis. 
# #### LMAOO
logging.info(response1, response2)