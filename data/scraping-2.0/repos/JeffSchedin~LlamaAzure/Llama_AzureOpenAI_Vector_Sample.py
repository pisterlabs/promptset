# November 2023 - Llama Index Vector Store Example
# Here is a revision of the Llama Index Vector Store example, but using Azure OpenAI instead of the default OpenAI API
# Make sure you have an Azure OpenAI account and have deployed your own models! 

import openai
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding

# In a 'real' setting, you would get these from your Key Valut or app configuration 

openai.api_type = "azure"
openai.api_base = https://<your path>.openai.azure.com/
openai.api_version = "<your version>"
openai.api_key = "<insert your key>"
model: str = "text-embedding-ada-002"

# Configure the Azure OpenAI API
llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="gpt-35-turbo-16k",
    api_key=openai.api_key,
    azure_endpoint=openai.api_base,
    api_version=openai.api_version
)

# In Azure, you need to deploy your own embedding model as well as your own chat completion model
# Currently, with Azure OpenAI, the text-embedding-ada-002 model is the one to use.

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="<your deployment name>",
    api_key=openai.api_key,
    azure_endpoint=openai.api_base,
    api_version=openai.api_version
)

# Need to tell Llama what models to use
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# Process the Docs to the Vector - change 'data' to match your folder/path name 
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True,save_path="vector_store.bin")

# Now confgiure your quey engine to use our llm & embed models
query_engine = index.as_query_engine(service_context=service_context)

# From the Sample
#response = query_engine.query("What are the new Software Features in Cisco IOS XE 17.12.1a?")
#print(response)

# Just for demo purposes, we will loop and ask the user for input - you should really use the query_engine.chat if you want to chat with the bot
while True:
    user_input = input("User: ")
    response = query_engine.query(user_input)  
    print(response)
