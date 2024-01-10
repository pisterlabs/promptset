from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI

# load env variables
load_dotenv(find_dotenv())



llm = OpenAI(model_name='text-davinci-003')
# print(llm("explain large language models in one sentence"))


# wrappers
# import a schema;
from langchain.schema import(
    AIMessage,
    HumanMessage, # human message is the user message
    SystemMessage # system message is what you use to configure the system
)
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.3)
messages = [
    SystemMessage(content="You are an expert data scientist"),
    HumanMessage(content="Write a python script that trains a neural network on simulated data")
]
# you use this as an input to the chat model
# response=chat(messages)
# print(response)



# prompts - what we're going to send to our LLM
"""
prompts are likely going to be dynamic as they are used in applications.
For this, Langchain has PromptTemplates.
Allows us  to take a piece of text and inject a user input into that text -> then format the prompt and 
feed it into the langauge model.
"""

template = """
You are an expert data scientist with an expertise in building deep learning models.
Explain the concept of {concept} in a couple of lines.
"""
# templates
from langchain import PromptTemplate
prompt = PromptTemplate(
    input_variables=['concept'],
    template=template,
)

# a basic example of changing the prompt with user input.
# print(llm(prompt.format(concept='autoencoder')))



# chains
"""
A chain takes a language model and a prompt template and combines them into an interface
that takes an input from the user and outputs an answer from the language model.
Like a composite function where the inner is the prompt template and outer function is the language model.

Also we can build sequential chains where we have one chain returning an output and then a 
second chain taking the output from the first chain as an input.
"""
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
chain = LLMChain(llm=llm, prompt=prompt)
# print(chain.run("autoencoder"))

second_prompt = PromptTemplate(
    input_variables=['ml_concept'],
    template = "Turn the concept description of {ml_concept} and explain it to me like I'm five years old, in 500 words.",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

# Run the chain specifying only the input variable for the first chain.
explanation = overall_chain.run('Autoencoder')
print(explanation)



# embedding and vector stores
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
)
texts = text_splitter.create_documents([explanation])
# print(texts[0].page_content)
embeddings = OpenAIEmbeddings(model_name='ada')


import os
import pinecone 
from langchain.vectorstores import Pinecone
 
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENV'),
)


index_name = 'langchain-tester'
index=pinecone.Index('langchain-tester')
model_name = 'gpt-3.5-turbo'
search = Pinecone.from_documents(texts, embeddings, index_name=index_name)
query = "What is magical about an autoencoder?"
result = search.similarity_search(query)

print(result)


# agent
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI

agent_executor = create_python_agent(
    llm=OpenAI(temperature=0, max_tokens=1000),
    tool=PythonREPLTool(),
    verbose=True
)
agent_executor.run("Find the roots (zeros) if quadratic function 3 * x**2 + 2*x - 1")






