from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from wonderwords import RandomWord

llm = Ollama(model="orca-mini")
embeddings = OllamaEmbeddings(model="orca-mini")


def batch_query_ollama(batch_of_prompts):
    return llm.batch(batch_of_prompts)

# Ollama Embeddings creation demo
# sample_text = "This is an example text to turn into a vector"
# query_result = embeddings.embed_query(sample_text)
# print(query_result)


# Example usage:
# movie_info = ollama_generation()
# print("Movie information:", movie_info)



