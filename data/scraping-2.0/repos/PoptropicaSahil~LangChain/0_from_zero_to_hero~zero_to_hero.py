import warnings


# Filter out UserWarnings - should come before the warning causing thing
warnings.filterwarnings("ignore", 
                        category=UserWarning
                        )


from dotenv import load_dotenv
import os
import logging 

from langchain.llms import OpenAI



# Logging Configuration
logging.basicConfig(filename='langchain.log', 
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# load env variables
load_dotenv('.env')

# load API key from environment
openai_key = os.getenv("OPENAI_API_KEY")
activeloop_key = os.getenv("ACTIVELOOP_TOKEN")
activeloop_org_id = os.getenv("ACTIVELOOP_ORG_ID")


# logging.INFO is an integer in itself
# use logging.info instead
# logging.info(f"OpenAI key is {openai_key}")
logging.info("hi there check hello")


######################################################
######################################################
### Basic Prompting

# llm = OpenAI(model_name="text-davinci-003", temperature=0.9)

# # text = "What would be a good company name for a company that makes colorful socks?"
# text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."

# result = llm(text)
# logging.info(result)
# print(result)


######################################################
######################################################
# The Chains

# from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI
# from langchain.chains import LLMChain

# llm = OpenAI(model="text-davinci-003", temperature=0.9)

# prompt = PromptTemplate(
#     input_variables = ["product"],
#     template = "What is a good name for a company that makes {product}?",
# )

# chain = LLMChain(llm=llm, prompt=prompt)

# # Run the chain only specifying the input variable.
# # result = chain.run("aluminium and copper wires")

# user_input = input("What type of product ")
# # user_input = "aluminium and copper wires"
# result = chain.run(user_input)

# # result = chain.run(input = "What type of product ") # does not work

# logging.info(result)
# print(result)


######################################################
###################################################### 
### Memory

# from langchain.llms import OpenAI
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory

# llm = OpenAI(model="text-davinci-003", temperature=0)

# # Memory, such as ConversationBufferMemory, 
# # acts as a wrapper around ChatMessageHistory, 
# # extracting the messages and providing them to the chain for better context-aware generation.

# conversation = ConversationChain(
#     llm=llm,
#     verbose=True,
#     memory=ConversationBufferMemory()
# )

# # Start the conversation
# conversation.predict(input="Tell me about yourself.")

# # Continue the conversation
# conversation.predict(input="What can you do?")
# conversation.predict(input="How can you help me with data analysis?")

# logging.info(conversation)

# # Display the conversation
# print(conversation)



######################################################
###################################################### 
### DeepLake Vector Store

# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import DeepLake
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms import OpenAI
# from langchain.chains import RetrievalQA

# # Before executing the following code, make sure to have your
# # Activeloop key saved in the “ACTIVELOOP_TOKEN” environment variable.

# logging.info("Running for DeepLake Vector Store")

# # instantiate the LLM and embeddings models
# llm = OpenAI(model="text-davinci-003", temperature=0)
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

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
# my_activeloop_org_id = activeloop_org_id 
# my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
# dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
# db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# # add documents to our Deep Lake dataset
# db.add_documents(docs)

# logging.info("Done adding embeddings to DeepLake path")



######################################################
###################################################### 
### DeepLake Vector Store - Retrieval
# Did not work out

from langchain.vectorstores import DeepLake
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType


logging.info("Running for Retriving DeepLake Vector Store")


llm = OpenAI(model="text-davinci-003", temperature=0)
# db = DeepLake.deeplake.load('hub://sahilg/langchain_course_from_zero_to_hero')

# create a RetrievalQA chain
retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm,
	chain_type="stuff",
	retriever=db.as_retriever()
)

tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

# let's create an agent that uses the RetrievalQA chain as a tool
agent = initialize_agent(
	tools = tools,
	llm = llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)

logging.info("Initialised the agent for Retreival QA")


logging.info("Asking question to  Agent")

response = agent.run("When was Napoleone born?")
print(response)