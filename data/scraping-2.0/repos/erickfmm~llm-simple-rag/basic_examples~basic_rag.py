
def make_splits():
    from langchain.document_loaders import WebBaseLoader

    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    data = loader.load()

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
    return all_splits

def make_vectorstore():
    all_splits = make_splits()
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=HuggingFaceEmbeddings())
    return vectorstore


if __name__ == "__main__":
    from langchain.callbacks.manager import CallbackManager
    from langchain.llms import LlamaCpp
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    #https://api.python.langchain.com/en/stable/llms/langchain.llms.llamacpp.LlamaCpp.html
    llm = LlamaCpp(
        model_path="/usr/src/app/models/luna-ai-llama2-uncensored.Q2_K.gguf",
        temperature=0.75,
        max_tokens=4098,
        n_ctx=4098,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    # MY CODE:
    vectorstore = make_vectorstore()
    ## END

    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    # Prompt
    prompt = PromptTemplate.from_template(
        "Summarize the main themes in these retrieved docs: {docs}"
    )

    # Chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Run
    question = "What are the approaches to Task Decomposition?"
    docs = vectorstore.similarity_search(question, k=2, fetch_k=5)
    result = llm_chain(docs)

    # Output
    print(result["text"])