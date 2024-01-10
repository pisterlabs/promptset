import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))


def load_documents(file_name):
    name, ext = os.path.splitext(file_name)
    if ext == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f"Loading {file_name}")
        loader = PyPDFLoader(file_name)
    else:
        print("Document format not supported!")
        return None
    _pages = loader.load()
    return _pages


def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data


def split_pages(pages, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(pages)
    return chunks


def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total tokens: {total_tokens}')
    print(f'Embedding cost in USD: {total_tokens / 1000 * 0.0004:.6f}')


class PineconeActivities:
    def __init__(self):

        self.embeddings = OpenAIEmbeddings()

    def insert_or_fetch_embeddings(self, index_name, chunks=None):
        if index_name in pinecone.list_indexes():
            print(f'Index {index_name} already exists')
            vector_store = Pinecone.from_existing_index(index_name, self.embeddings)
            print('Ok')
        else:
            print(f'Creating index {index_name} and embeddings ...', end='')
            pinecone.create_index(index_name, dimension=1536, metric='cosine')
            vector_store = Pinecone.from_documents(chunks, self.embeddings, index_name=index_name)
            print('Ok')

        return vector_store

    def delete_pinecone_index(self, index_name='all'):
        if index_name == 'all':
            indexes = pinecone.list_indexes()
            print('Deleting all indexes ... ')
            for index in indexes:
                pinecone.delete_index(index)
            print('Ok')
        else:
            print(f'Deleting index {index_name} ...', end='')
            pinecone.delete_index(index_name)
            print('Ok')


def ask_and_get_answer(vector_store, q):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer


def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    return result, chat_history


def ask_ui_with_history(_vector_store):
    import time
    i = 1

    chat_history = []

    print("Write Quit or Exit to quit")
    while True:
        q = input(f"Question #{i}")
        i = i + 1
        if q.lower() in ["quit", "exit"]:
            print("Quiting")
            time.sleep(2)
            break
        result, _ = ask_with_memory(_vector_store, q, chat_history)
        print(result['answer'])
        print("----------------------------------------------------------------------")


if __name__ == '__main__':
    # pages = load_documents('files/BBI_ENGLISH_VERSION.pdf')
    # chunks = split_pages(pages)
    # print(chunks[10].page_content)
    # print_embedding_cost(chunks)
    pine = PineconeActivities()
    # pine.delete_pinecone_index()
    vector_store = pine.insert_or_fetch_embeddings('uhuru-bbi')
    # print(ask_and_get_answer(vector_store, "How do they plan to resolve election issues?"))
    ask_ui_with_history(vector_store)