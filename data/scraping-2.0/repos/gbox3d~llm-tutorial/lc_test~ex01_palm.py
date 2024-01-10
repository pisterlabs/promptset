#%%
import os 
from dotenv import load_dotenv, find_dotenv

load_dotenv() # read local .env file

# configure palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm

#%%
llm = GooglePalm()

llm.invoke("earth's satellite is?")


# %%
