from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

from core.retriever import QuestionRetreiver


class FAQAssistant:
    def __init__(
        self,
        embeddings_file_path: str = "static/FAQ.pkl",
        faq_file_path: str = "static/FAQ.json",
    ):
        self.retriever = QuestionRetreiver(
            embeddings_file_path=embeddings_file_path,
            faq_file_path=faq_file_path,
            k=50,
            max_unique_questions=7,
        )
        self.qa_chain = load_qa_chain(
            ChatOpenAI(temperature=0.1), chain_type="stuff", verbose=True
        )
        self.qa = RetrievalQA(
            combine_documents_chain=self.qa_chain,
            retriever=self.retriever,
            verbose=True,
            return_source_documents=True,
        )

    def answer(self, question: str):
        result = self.qa({"query": question})
        return result["result"]
