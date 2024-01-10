import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import LatexTextSplitter
from langchain.vectorstores.utils import DistanceStrategy
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from elasticsearch import Elasticsearch
import os
import weaviate

import json
import requests


from docs_utils import get_academy_docs
from indexing_elasticsearch import es_indexing, es_search
from indexing_huggingface import indexing_huggingface, search_huggingface
from models.main import bge_large_en_start, multilingual_e5_large, openai_start
from retrievers import get_weaviate_openai_retriever, get_weaviate_huggingface_retriever, weaviate_search
import langchain

langchain.debug = True
langchain.verbose = True

questions = [
    "How install global search in creatio?",
    "How can I switch Classic UI to Freedom?",
    "is possible for us set up an external identity system, not ADSF or SignOne? The provider is using SAML 2.0",
    "Could you please give us instructions(how to use it and configure) regarding the chatGPT.",
    "Is it possible to Remove/Edit/Hide certain fields from appearing in the Opportunity Timeline?",
    "our onprem customer DMH wants to connect their Creatio to cloud Exchange (MS365). The admin asked, which ports we need in the firewall to do that?",
    "Де налаштовуються сповіщення користувачам? Сповіщення про нове завдання на них, новий лід, повідомлення в стрічці з новинами компанії, тощо.",
    "Як в Журналі змін об'єкта дізнатися які саме поля правились?",
    "Як налаштувати авторизацію користувачів в додатку Creatio через Microsoft365 (Azure Active Directory)",
    "Як можна додати маску для телефонного номеру на Classic UI?",
    "Как изменить назвние раздела?",
    "How to configure mobile notification in creatio? Please advise, thanks a lot",
    "При спробі додати Фейсбук або лінкедін як засіб зв'язку контрагента система видає вікно авторизаці фейсбук/лінкедін. Це так має бути?",
    "What is the difference between Classic and Freedom UI?",
]

def translate_to_en(query):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = "I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it in English. I want you to only reply the correction, the improvements and nothing else, do not write explanations."
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
    )
    return completion.choices[0].message.content

def elastic_hybrid_search(query="How install global search in creatio?"):
    embedding = OpenAIEmbeddings()
    client = Elasticsearch(
        # hosts=["http://localhost:9200"],
        cloud_id="academy:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGRiM2MzNWE3NDJjNDRhZGE4OGY4YjYwOTUxOWM0YTg4JGFhZThmMWM1NTdhYTRmMTZhYTY5YzlkNzg2YzRmYTM4",
        request_timeout=280,
        retry_on_timeout=True,
        max_retries=2,
        basic_auth=("elastic", "bbOUCV7nDnK8r05DjtRz5zfj")
    )
    client.options(
        request_timeout=280
    ).cluster.health(
        timeout="280s",
        master_timeout="280s",
    )
    # store = ElasticsearchStore.from_documents(
    #     documents=splits_documents,
    #     embedding=embedding,
    #     index_name="academy-from-md-latex-text",
    #     vector_query_field="embedding",
    #     strategy=ElasticsearchStore.ApproxRetrievalStrategy(
    #         hybrid=True,
    #     ),
    #     distance_strategy="COSINE",
    #     es_connection=client
    # )
    # store.client.indices.refresh(index="academy-from-md-latex-text")
    store = ElasticsearchStore(
        embedding=embedding,
        index_name="academy-from-md-latex-text",
        vector_query_field="embedding",
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(
            hybrid=True,
        ),
        distance_strategy=DistanceStrategy.COSINE,
        es_connection=client
    )
    results = store.similarity_search(query)
    print("start ES hybrid search by query: ", query)
    print(list(map(lambda x: x.metadata["source"], results)))
    print("\n")

def weaviate_hybrid_search(query="How install global search in creatio?"):
    retriever = get_weaviate_openai_retriever()
    # print("start indexing documents")
    # store.add_documents(docs)
    print("start weaviate hybrid search by query: ", query)
    results = retriever.get_relevant_documents(
        query,
        score=True,
    )
    print(list(map(lambda x: x.metadata["source"], results)))
    return results, retriever

    # query = "Как установить глобальний поиск?"
    # print("start hybrid search result by query: ", query)
    # results = store.get_relevant_documents(
    #     query,
    #     score=True,
    # )
    # print(list(map(lambda x: x.metadata["source"], results)))


def generate_weaviate_answer(question: str):
    retriever = get_weaviate_openai_retriever()
    template = """You are Creation platform bot.
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use code fragments and keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
    )
    result = qa_chain({"query": question})
    print(result["result"])
    file_names = list(map(lambda x: x.metadata["source"], result["source_documents"]))
    print(file_names)
    return result["result"]

def print_hi():
    # docs = get_academy_docs()
    # elastic_hybrid_search()
    #query = "How start creatio in docker?"
    #query = "How install global search in creatio?"
    for q in questions:
        print('\n')
        print('---------------------------')
        elastic_hybrid_search(query=q)
        # elastic_hybrid_search(query=en_query)
        # weaviate_hybrid_search()
        weaviate_hybrid_search(query=q)
    # en_query = translate_to_en(query)
    # elastic_hybrid_search(query=en_query)
    # weaviate_hybrid_search()
    # docs, retriever = weaviate_hybrid_search(query=en_query)
    #generate_weaviate_answer(query)
    # pinecone.init(
    #     api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    #     environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    # )
    #
    # index_name = "academy"

    # First, check if our index already exists. If it doesn't, we create it
    # if index_name not in pinecone.list_indexes():
    #     # we create a new index
    #     pinecone.create_index(
    #         name=index_name,
    #         metric='cosine',
    #         dimension=1536
    #     )
    # embedding = OpenAIEmbeddings()
    # docsearch = Pinecone.from_documents(docs, embedding, index_name=index_name)

    # if you already have an index, you can load it like this
    # docsearch = Pinecone.from_existing_index(index_name, embeddings)

    # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # store = Chroma.from_documents(splits_documents, embedding_function)

    # html2text = Html2TextTransformer()
    # docs_transformed = html2text.transform_documents(docs)
    # print(docs_transformed)
    # markdown_text = markdownify.markdownify(docs[0].page_content)
    # print(docs)
    #
    # # loader1 = TextLoader("gs.md", encoding="utf-8")
    # # docs1 = loader1.load()
    # # print(docs1)

    # text_splitter = MarkdownTextSplitter()
    # docssplit = text_splitter.split_documents(docs1)
    # print(docssplit)

# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    bge_large_en_start()
    multilingual_e5_large()
    openai_start()
