from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

if __name__ == '__main__':
    loader = TextLoader('./2020_state_of_the_union.txt', encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)

    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    store = Chroma.from_documents(texts, embeddings, collection_name="2020_state_of_the_union")

    llm = OpenAI(temperature=0)
    chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())

    while True:
        my_input = input("> ")
        print(chain.run(my_input))
