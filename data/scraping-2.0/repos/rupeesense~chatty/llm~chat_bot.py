from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from context_store.llm_logger import llm_logger
from llm.llm_factory import LLMFactory

PROMPT = '''<s>[INST]You are a helpful personal finance assistant that translates that answers user queries respectfully.
Try to explain the numbers in a helpful manner. You have to report all numbers in rupees.

{history}
Human: {text}
Context: {context}
AI: [/INST]'''


class Chatty:

    def __init__(self):
        self._llm = LLMFactory().get_chat_llm()
        custom_template = PromptTemplate(input_variables=["history", "text", "context"], template=PROMPT)
        self.llm_chain = LLMChain(prompt=custom_template, llm=self._llm, verbose=True)

    def respond(self, message: str, context: str, chat_history: str) -> str:
        if chat_history is not "":
            chat_history = "history of conversation: \n" + chat_history
        model_response = self.llm_chain.run(text=message, context=context, history=chat_history)
        llm_logger.log(use_case='chatty', prompt=message, response=model_response, llm_params=self._llm._default_params)
        return model_response
