import time
from models.base_model import BaseLLM, output_parser
from langchain.llms import OpenAI
from keys.keys import openai_key
import tiktoken


class Openai(BaseLLM):
    def __init__(self) -> None:
        super().__init__()
        self.set_model()
        self.set_tokenizer()
        
    def set_model(self)->None:
        self.model_id = 'OpenAI'
        self.model = OpenAI(openai_api_key=openai_key, model="text-davinci-003", temperature=0)

    def set_tokenizer(self)->None:
        self.tokenizer = tiktoken.encoding_for_model("text-davinci-003")

    def run(self, prompt)->str:
        if self.model is None:
            self.model = OpenAI(openai_api_key=openai_key, model="text-davinci-003", temperature=0) # latest model (2023-07-21)
            self.tokenizer = tiktoken.encoding_for_model("text-davinci-003")
            
        # inference
        self.tokens = len(self.tokenizer.encode(prompt))
        start_time = time.time()
        generated = self.model(prompt)

        # elapsed time
        self.calculate_elapsed_time(start_time=start_time)

        # decode
        return output_parser(generated)
