from langchain.chains import RetrievalQA

class QAChain:
    def __init__(self, llm, chain_type, retriever):
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever.as_retriever(),
        )

    def get_response(self, query):
        response = self.qa_chain(query)
        return response["result"]