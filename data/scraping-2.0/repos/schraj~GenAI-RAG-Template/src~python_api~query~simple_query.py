from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings
from python_api.shared.app_base import initialize_openai

def query():
    initialize_openai()

    embeddingsModel = OpenAIEmbeddings(
        model='text-embedding-ada-002')

    db = Milvus(embedding_function=embeddingsModel)
    retriever = db.as_retriever(search_kwargs={"k": 16})
    chat_model = ChatOpenAI(
        engine="gpt-35-turbo",
    )
    qa = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=retriever, return_source_documents=True)
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        res = qa(query)
        answer, docs = res['result'], res['source_documents']

        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer:")
        print(answer)

        # for document in docs:
        #     print("\n> " + document.metadata["source"] + ":")
        #     print(document.page_content)

