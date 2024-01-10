# Load web page
import argparse

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embed and store
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings import OllamaEmbeddings # We can also try Ollama embeddings

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def main():
    parser = argparse.ArgumentParser(description='Filter out URL argument')
    parser.add_argument('--url', type=str, default='http://example.com', required=True, help='The URL to filter')

    args = parser.parse_args()
    url = args.url
    print(f"Using URL: {url}")

    loader = WebBaseLoader(url)
    data = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
    print(f"Split into {len(all_splits)} chunks")

    # Store config
    vectorstore = Chroma.from_documents(documents=all_splits,
                                        embedding=GPT4AllEmbeddings())
    
    # Retrieve
    # question = "Sobre qué trata el artículo {url}?"
    # docs = vectorstore.retrieve(question)

    print(f"Loaded {len(data)} documents from {url}")
    # print(f"Retrieved {len(docs)} documents from {question}")

    # RAG prompt
    from langchain import hub
    QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

    # LLM
    llm = Ollama(model="llama2",
                 verbose=True,
                 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    
    # QA Chain
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}

    )

    # Ask a question
    question = "Haz un resumen sobre los que hay en la dirección web {url}?"
    result = qa_chain({"query": question})

    print(result)

if __name__ == "__main__":
    main()