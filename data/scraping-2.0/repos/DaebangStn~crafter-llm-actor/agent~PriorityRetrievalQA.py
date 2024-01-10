from typing import List

from langchain import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever


class PriorityRetrievalQA:
    llm: BaseLanguageModel
    retrieverList: list = []
    prompt: str = """Do not use your background knowledge and answer only based on the information given next. 
    Remember, you should only answer from the context given. Answer only for questions within the context provided.
    don't try to make up an answer.
    
    CONTEXT:
    {context}
    
    Topic: {question}

    Must Follow: If you can't determine the correct answer with the information you have, say "don't know".

    Strategy: Imagine that you are an expert player who thinks strategically.
    
    Give me ALL relevant information in context about given topic:"""

    chains: list = []

    def __init__(self, llm: BaseLanguageModel, retrieverList: List[BaseRetriever]):
        self.llm = llm
        self.retrieverList = retrieverList

        assert llm is not None, "Must have a language model"
        assert len(self.retrieverList) > 0, "Must have at least one retriever"

        _prompt = PromptTemplate(
            template=self.prompt, input_variables=["context", "question"]
        )

        chain_type_kwargs: dict = {"prompt": _prompt}

        for retriever in self.retrieverList:
            chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs,
                verbose=True,
            )
            self.chains.append(chain)

    def query(self, query: str):
        for chain in self.chains:
            response = chain({"query": query})
            if response["result"].lower().find("don't know") == -1:
                # TODO add a validation step with LLM here (answer and document)
                return response
            else:
                print("DN")

        return None
