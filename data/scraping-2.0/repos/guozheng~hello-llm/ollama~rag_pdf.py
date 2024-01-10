from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain import hub
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def main():
    MODEL_PROMPT_MAPPINGS = {
        "llama2-uncensored": "rlm/rag-prompt-llama",
        "llama2": "rlm/rag-prompt-llama",
        "mistral": "rlm/rag-prompt-mistral",
    }

    MODEL_NAME = "llama2-uncensored"
    PROMPT_NAME = MODEL_PROMPT_MAPPINGS[MODEL_NAME]

    # retrieve data from PDF and split
    loader = PyPDFLoader("../data/Prompt_Engineering_For_ChatGPT_A_Quick_Guide_To_Te.pdf")
    all_splits = loader.load_and_split()
    # print(all_splits[0])

    # embed chunks and store in vectorstore
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

    # retrieve
    question = "What are the advantages of using a prompt engineering approach?"
    # docs = vectorstore.similarity_search(question)
    # print("Retrieved documents:")
    # print(docs)

    # QA chain
    QA_CHAIN_PROMPT = hub.pull(PROMPT_NAME)
    ollama = Ollama(model=MODEL_NAME, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True)
    qa_chain = RetrievalQA.from_chain_type(
        ollama,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    result = qa_chain({"query": question})
    print("Answer:")
    print(result)

# generate main function
if __name__ == "__main__":
    main()