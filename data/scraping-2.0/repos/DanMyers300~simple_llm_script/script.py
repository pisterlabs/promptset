from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings

# Set up the Ollama language model and callback manager
llm = Ollama(
    base_url="https://d9b64cd7ae64544fb87411ff82300ef1e.clg07azjl.paperspacegradient.com", 
    model="llama2", 
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# Set up the OllamaEmbeddings model
oembed = OllamaEmbeddings(
    base_url="https://d9b64cd7ae64544fb87411ff82300ef1e.clg07azjl.paperspacegradient.com", 
    model="llama2"
)

# Ask the user for a query
user_query = input("Enter your query: ")

# Use OllamaEmbeddings to embed the user's query
oembed.embed_query(user_query)

# Use the Ollama language model to generate a response
response = llm(user_query)

# Print the response
print(response)
