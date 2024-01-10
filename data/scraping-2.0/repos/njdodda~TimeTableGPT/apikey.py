import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers

def llmKeySetup():
    os.environ['OPENAI_API_KEY'] = 'Add your key'
    # Create instance of OpenAI LLM
    llm = OpenAI(temperature=0.1, verbose=True)
    embeddings = OpenAIEmbeddings()
    return llm,embeddings