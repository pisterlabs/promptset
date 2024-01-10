from typing import Optional

from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

from chatytt.conf.config import load_config
from chatytt.chains.base_chain import BaseChain
from chatytt.vector_store.pinecone_db import PineconeDB

chain_conf = load_config()["chains"]


class QAChain(BaseChain):
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        chain_type: str = "stuff",
    ):
        super().__init__()
        self.model_name = model_name if model_name else chain_conf["model_name"]
        self.temperature = temperature
        self.llm = OpenAI()
        self.chain = load_qa_chain(llm=self.llm, chain_type=chain_type)

    def get_response(self, query: str, context: Optional[str]):
        answer = self.chain.run(input_documents=context, question=query)

        return answer


if __name__ == "__main__":
    load_dotenv()

    pinecone_db = PineconeDB(
        index_name="youtube-transcripts", embedding_source="open-ai"
    )

    query = (
        "Is buying a house a good financial decision to make in your 20s ? Give details on the "
        "reasoning behind your answer. Also provide the name of the speaker in the provided context from"
        "which you have constructed your answer."
    )
    similar_docs = pinecone_db.get_similar_docs(query=query)

    qa_chain = QAChain()
    response = qa_chain.get_response(query=query, context=similar_docs)
    print(response)
