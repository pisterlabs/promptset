import os

def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == ".pdf":
        from langchain.document_loaders import PyPDFLoader
        print(f"Loading {file}")
        loader = PyPDFLoader(file)

    elif extension == ".docx":
        from langchain.document_loaders import Docx2txtLoader
        print(f"Loading {file}")
        loader = Docx2txtLoader(file)

    else:
        return None

    data = loader.load()
    return data


def load_from_wikipedia(query, lang='en'):
    from langchain.document_loaders import WikipediaLoader

    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=10)

    data = loader.load()

    return data

def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

import tiktoken
def printing_cost(texts):
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f"Total Tokens is {total_tokens}")
    print(f"Embedding cost in USD {total_tokens / 1000 * 0.0004:0.6f}")


def insert_of_fetch_embeddings(index_name):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()

    pinecone.init(
        api_key=os.environ.get('PINECODE_API_KEY'),
        environment=os.environ.get("PINECONE_ENV"))

    if index_name in pinecone.list_indexes():
        print(f"Index {index_name} already exists")
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
    else:
        print(f"Creating index name")
        pinecone.create_index(index_name, dimension=1536,
                              metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print("ok")

    return vector_store


def delete_index(index_name='all'):
    import pinecone
    pinecone.init(
        api_key=os.environ.get('PINECODE_API_KEY'),
        environment=os.environ.get("PINECONE_ENV"))
    if index_name == "all":
        indexes = pinecone.list_indexes()
        print("Deleting All indexes ..")
        for index in indexes:
            pinecone.delete_index(index)
    else:

        print(f"Deleting Index {index_name}")
        pinecone.delete_index(index_name)


def ask_get_answer(vector_store, question):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    answer = chain.run(question)

    return answer


def ask_with_memory(vector_store, question, chat_histoy):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-4-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 3})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)

    result = crc({"question": question, "chat_history": chat_histoy})

    chat_history.append((question, result['answer']))

    return result, chat_history
