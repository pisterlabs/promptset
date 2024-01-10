from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

from langchain_lit import load_txt_documents, load_markdown_documents
from llm_utils import Retrieval
from llm_utils import ConversationalRetrievalChainAgent
from llm_utils import LlmEmbedding
from qdrant_lit import QdrantVectorStore


def load_llm_model():
    model_name = "TheBloke_Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    # model_name = "TheBloke_Mistral-7B-OpenOrca-GGUF/mistral-7b-openorca.Q4_K_M.gguf"
    llm = LlamaCpp(
        model_path=f"../models/{model_name}",
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        n_ctx=2048,  # 請求上下文 ValueError: Requested tokens (1130) exceed context window of 512
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        streaming=True,
        n_gpu_layers=52,
        n_threads=4,
    )
    return llm


class LlmQaChat:
    def __init__(self, llm, vector_db, retrieval: Retrieval = None):
        self.llm = llm
        self.vector_db = vector_db
        if retrieval is None:
            retrieval = Retrieval(vector_db, llm, llm_embedding=vector_db.llm_embedding)
        self.retrieval = retrieval

    def ask(self, question: str):
        retriever = self.retrieval.get_retriever("sample1")
        llm_qa = ConversationalRetrievalChainAgent(self.llm, retriever)
        return llm_qa.ask(question)

    def ask_retriever(self, retriever, question: str):
        llm_qa = ConversationalRetrievalChainAgent(self.llm, retriever)
        # llm_qa = RetrievalQAAgent(self.llm, retriever)
        return llm_qa.ask(question)


def main():
    llm_embedding = LlmEmbedding("../models/BAAI_bge-base-en")
    docs1 = load_txt_documents("../data")
    docs2 = load_markdown_documents("../data")

    def load_markdown_documents(data_path: str):
        md_loader = DirectoryLoader(data_path, glob='*.md', loader_cls=UnstructuredMarkdownLoader)
        return md_loader.load()

    print("loading llm")
    llm = load_llm_model()
    print("llm done")

    vector_db = QdrantVectorStore(llm_embedding)
    vector_db.open(url="http://localhost:6333")
    retriever = Retrieval(vector_db, llm, llm_embedding)

    # all_collections = vector_db.get_all_collections()
    # print(f"{all_collections=}")

    # vector_db.recreate_collection('sample1')
    # retriever.add_parent_document('sample1', docs1)
    # retriever.add_parent_document('sample1', docs2)
    # retriever.add_parent_document('sample2', docs2)
    print(f"add documents done")

    query = "How to convert a B2B2C domain to a B2C domain?"
    print(f"{query=}")
    llm_qa_chat = LlmQaChat(llm, vector_db)
    # result1 = vector_db.search('sample1', query)
    # print("===")
    # print(f"{result1=}")

    # worked
    # answer = llm_qa_chat.ask(query)
    # print("===============================================")
    # print(answer)

    answer = llm_qa_chat.ask(query)
    print("===========================================")
    print(f"{answer=}")

    # qa = retriever.get_parent_document_retriever_qa('sample1')
    # qa = retriever.merge_parent_document_retriever_qa(['sample1'])
    # print("merge_parent_document_retriever_qa query...")
    # result = qa.run(query)
    # print("")
    # print("")
    # print("---------------")
    # print(f"{result=}")

    # retriever.create_collection('sample')
    # retriever.upsert_docs('sample', docs)
    # result = retriever.search('sample', 'How to create pinia store in vue3?')
    # print(f"{result=}")


if __name__ == '__main__':
    main()
