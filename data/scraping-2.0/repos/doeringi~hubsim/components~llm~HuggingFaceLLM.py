from langchain.base_language import BaseLanguageModel
from langchain.llms import HuggingFacePipeline
from langchain.llms.fake import FakeListLLM
from transformers import pipeline, AutoTokenizer
import os
from transformers import AutoModelForCausalLM
from components.llm.AbstractLLM import AbstractLLM


class HuggingFaceLLM(AbstractLLM):
    model_id: str
    tokenizer: AutoTokenizer
    llm: BaseLanguageModel
    tokenizer_arguments: dict
    model_arguments: dict

    def __init__(self):
        self.tokenizer_arguments = {}
        self.model_arguments = {
            "temperature": 0.9,
            "max_new_tokens": 50,
        }

    def download_llm(self, access_token=None):
        model_id = self.get_model_id
        
        model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token)
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)

        model.save_pretrained(os.path.join("models", model_id))
        tokenizer.save_pretrained(os.path.join("models", model_id))
        print("Downloaded LLM")


    def load_llm(self) -> BaseLanguageModel:
        model_id = self.get_model_id

        model = AutoModelForCausalLM.from_pretrained(os.path.join("models", model_id))
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join("models", model_id))

        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=self.tokenizer,
            model_kwargs=self.get_model_arguments,
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)
