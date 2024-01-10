import time
from models.base_model import BaseLLM, output_parser
from langchain.chat_models import ChatOpenAI
from keys.keys import openai_key
import tiktoken
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

class ChatOpenai(BaseLLM):
    def __init__(self) -> None:
        super().__init__()
        self.set_model()
        self.set_tokenizer()
        
    def set_model(self)->None:
        self.model_id = 'ChatOpenAI'
        self.model = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo", temperature=0) 

    def set_tokenizer(self)->None:
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, prompt)->str:
        if self.model is None:
            self.model = ChatOpenAI(openai_api_key=openai_key, model="gpt-3.5-turbo", temperature=0)
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            
        # inference
        self.tokens = len(self.tokenizer.encode(prompt))
        start_time = time.time()
        messages = [SystemMessage(content="You are a smart assistant."),
                    HumanMessage(content=prompt)]
        output = self.model(messages=messages)

        # elapsed time
        self.calculate_elapsed_time(start_time=start_time)

        # decode
        generated = output.content
        return output_parser(generated)
