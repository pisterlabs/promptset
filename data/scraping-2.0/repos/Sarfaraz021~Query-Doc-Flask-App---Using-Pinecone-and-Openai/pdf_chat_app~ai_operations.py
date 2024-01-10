from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

class AIOperations:
    def __init__(self, llm):
        self.chain = load_qa_chain(llm, chain_type="stuff")

    def run_chain(self, docs, query):
        return self.chain.run(input_documents=docs, question=query)
