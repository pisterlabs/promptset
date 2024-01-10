from langchain.chat_models.openai import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

class ChatbotGPT:
    def __init__(self,model: str, prompt: str, max_tokens:int, temperature: int, ) -> None:
        self.model = model
        self.temperature = temperature
        self.prompt = prompt
        self.llm = ChatOpenAI(model=self.model,
                         temperature= self.temperature,
                         max_tokens=max_tokens
                         )

    def load_chain(self, llm, prompt: str,chain_type: str = "stuff", verbose: bool = True, document_variable_name: str = "sources"):
        chain = load_qa_with_sources_chain(
            llm=llm,           
            chain_type=chain_type,
            verbose=verbose,
            prompt=prompt,
            document_variable_name=document_variable_name
        )

        return chain
    
    def question_answer(self, sources, query):
        chain = self.load_chain(self.llm, self.prompt)
        result = chain({"input_documents": sources, "question": query},
                       return_only_outputs=True)
        answer = result["output_text"]

        return answer
