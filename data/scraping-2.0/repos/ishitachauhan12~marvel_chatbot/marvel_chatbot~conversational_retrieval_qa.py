from langchain.document_loaders import TextLoader
from utils import AzureModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


class ConverationalRetrievalQaClass:
    def __init__(self):
        self.loader = TextLoader("./data.py")

    def prepare_data(self):
        documents = self.loader.load()
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()

        docsearch = FAISS.from_documents(chunks, embeddings)
        return docsearch

    def prepare_memory(self):
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        return memory

    def conversational_retrievalqa_chain(self, query):
        docsearch = self.prepare_data()
        memory = self.prepare_memory()
        llm_object = AzureModel()
        llm = llm_object.get_llm_model()
        qa = ConversationalRetrievalChain.from_llm(
            llm, docsearch.as_retriever(), memory=memory
        )
        response = qa.run(query)
        return response


if __name__ == "__main__":
    retrieval_qa_object = ConverationalRetrievalQaClass()
    query1 = "who is ceasar?"
    response1 = retrieval_qa_object.conversational_retrievalqa_chain(query1)

    print(response1)

    query2 = "was he a king?"
    response2 = retrieval_qa_object.conversational_retrievalqa_chain(query2)

    print(response2)
