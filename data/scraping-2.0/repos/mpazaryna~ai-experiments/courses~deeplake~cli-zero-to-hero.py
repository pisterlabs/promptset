import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


# instantiate the LLM and embeddings models
llm = OpenAI(model="text-davinci-003", temperature=0)

# Create Deep Lake dataset
# Use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "mpazaryna"
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

from langchain.agents import AgentType, Tool, initialize_agent

tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions.",
    ),
]

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

response = agent.run("When was Napoleone born?")
print(response)
