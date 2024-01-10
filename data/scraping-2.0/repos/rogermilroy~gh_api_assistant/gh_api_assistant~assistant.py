import time
import tiktoken
from gh_api_assistant.scrape import get_rest_api_data, split_api_data
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import FakeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser



def initialise_vectorstore(rest_api_data: str, vectorstore):
    # initialise the vectorstore with data respecting rate limits

    chunksize = 1000
    last_chunk_len = len(rest_api_data) % chunksize
    # chunks = [rest_api_data[i:i+1000] for i in range(0, len(rest_api_data), 1000)]
    batch_size = 0
    limit = 900_000
    texts = list()

    # TODO maybe overlap chunks to improve chance of chunk covering all it needs in one.
    #  change the size of step to lower than chunk size and figure out if there are side effects
    for i in range(0, len(rest_api_data)-last_chunk_len, chunksize):
        chunk = rest_api_data[i:i+chunksize]
        start = time.time()
        texts.append(chunk)
        batch_size+= len(chunk)
        if batch_size >= limit:
            print(f"adding texts {len(texts)}")
            vectorstore.add_texts(texts=texts)
            texts = list()
            batch_size = 0
            time_taken = int(time.time() - start)
            time.sleep(60 - time_taken)
    # deal with the final chunk
    last_chunk = rest_api_data[-last_chunk_len:]
    if len(texts) > 0:
        texts.append(last_chunk)
        print(f"Adding to texts new len: {len(texts)}")
    else:
        texts = [last_chunk]
    vectorstore.add_texts(texts=texts)


def get_assistant(api_key: str, collection_name: str):
    # TODO maybe split this into chunks based on semantics?

    vectorstore = Chroma(collection_name=collection_name, persist_directory="vectorstore", embedding_function=OpenAIEmbeddings(openai_api_key=api_key, show_progress_bar=True))

    # if the vectorstore collection hasn't been initialised yet then create and persist it
    elements_in_store = len(vectorstore.get()["ids"])
    print(elements_in_store)
    if elements_in_store == 0:
        print(f"Collection {collection_name} not initialised - fetching new source data")

        source_data = get_rest_api_data("https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json")
        initialise_vectorstore(rest_api_data=source_data, vectorstore=vectorstore)
        vectorstore.persist()
        print(len(vectorstore.get()["ids"]))
    else:
        print("Vectorstore already initialised.")

    # TODO check mmr kwargs - for more diversity of results
    # MMR kwargs fetch_k: default 20, lambda_mult default 0.5 1.0 min diversity, 0.0 max diversity
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5})

    template = """You are a helpful assistant, below is a question from a user and
    some relevant contexts which describe the Github REST API in openapi json format.
    Answer the question given the information in those contexts.
    If you cannot find the answer to the question, say "I'm afraid I can't answer that question.".

    Contexts:
    {context}

    Question: {question}

    Answer: """

    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(openai_api_key=api_key)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain
