from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def populate(vector_store):
    # is the store empty? find out with a probe search
    # (this is a hack, but it's the only way to find out)
    # (the vector store should have a method for this)
    # We're assuming a vector embedding of 4096 dimensions (for Ollama)
    # In case of a different embedding, change the dimension value.
    # For example, in case of OpenAI's text-embedding-ada-002, the value would be 1536.
    hits = vector_store.similarity_search_by_vector(
        embedding=[0.001] * 4096,
        k=1,
    )
    #
    if len(hits) == 0:
        # this seems a first run:
        # must populate the vector store
        # Load document source data
        loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        data = loader.load()
        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        # Add to vector store
        vector_store.add_documents(all_splits)
        return len(all_splits)
    else:
        return 0