from src.cove_chains import ChainOfVerification
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class ChainOfVerificationOpenAI(ChainOfVerification):
    def __init__(
        self, model_id, temperature, task, setting, questions, openai_access_token
    ):
        super().__init__(model_id, task, setting, questions)
        self.openai_access_token = openai_access_token
        self.temperature = temperature
        
        self.llm = ChatOpenAI(
            openai_api_key=openai_access_token,
            model_name=self.model_config.id,
            max_tokens=500
        )

    def call_llm(self, prompt: str, max_tokens: int) -> str:
        llm_chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        return llm_chain.invoke({})

    def process_prompt(self, prompt, _) -> str:
        # We do not need to do any processing here!
        return prompt