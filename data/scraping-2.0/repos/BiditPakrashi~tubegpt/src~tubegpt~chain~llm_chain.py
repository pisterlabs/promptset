from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

class LLMChain():
    def __init__(self) -> None:
        pass

    def create_chain(self,llm, prompt):
        self.chain  = LLMChain(
        llm=llm, prompt=prompt, verbose=True,)
        
    #fix this
    def query(self,question):
        response = self.chain.run(question)
        return response