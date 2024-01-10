#use dotenv to get the api keys from .env file
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")
#Import prompt and define PromptTemplate
from langchain import PromptTemplate
template = """You are an expert data scientist with an expertise in building deep learning models. 
Explain the concept of {concept} in a couple of lines"""
prompt = PromptTemplate(
    input_variables=["concept"],
    template=template,
)


#-------------------------------------------------------
#CHAIN CODE STARTS HERE
#Import LLMChain and define chain with language model and prompt as arguments.
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
#Run the chain only specifying the input variable.
# print(chain.run("autoencoder"))




#Define a second prompt 
second_prompt = PromptTemplate(
    input_variables=["ml_concept"],
    template="Turn the concept description of {ml_concept} and explain it to me like I'm five in 500 words",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)




#Define a sequential chain combining the two chains above: the second chain takes the output of the first chain as input
from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)
#Run the chain specifying only the input variable for the first chain.
explanation = overall_chain.run("autoencoder")
# print(explanation)




#--------------------------------------------
#EMBEDDINGS AND VECTOR STORES CODES STARTS HERE
#Import utility for splitting up texts and split up the explanation given above into document chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 0,
)
texts = text_splitter.create_documents([explanation])





#Turn chunks of texts into embeddings(vector representations of text) by importing OpenAI embeddings
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model_name="ada")




#Turn the first text chunk into a vector with the embedding
query_result = embeddings.embed_query(texts[0].page_content)
# print(query_result)



#Import and initialize Pinecone client
import os
import pinecone
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),  
    environment=os.getenv('PINECONE_ENV')  
)





#Upload vectors to Pinecone
index_name = "langchain-quickstart"
search = Pinecone.from_documents(texts, embeddings, index_name=index_name)



#Do a simple vector similarity search
query = "What is magical about an autoencoder?"
result = search.similarity_search(query)
print(result)