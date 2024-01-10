from langchain.base_language import BaseLanguageModel
from langchain.llms import HuggingFacePipeline
from langchain.llms.fake import FakeListLLM
from transformers import pipeline, AutoTokenizer
import os
from transformers import AutoModelForSeq2SeqLM
from components.llm.AbstractLLM import AbstractLLM


class LocalLLM(AbstractLLM):
    model_id: str
    tokenizer: AutoTokenizer
    llm: BaseLanguageModel
    tokenizer_arguments: dict
    model_arguments: dict

    def __init__(self, llm: BaseLanguageModel = None):
        self.model_id = "google/flan-t5-small"
        self.llm = llm
        self.tokenizer_arguments = {}
        self.model_arguments = {
            "temperature": 0.9,
            "max_new_tokens": 50,
        }

    @property
    def get_llm(self) -> BaseLanguageModel:
        return self.llm

    def download_llm(self):
        model_id = self.get_model_id

        if not os.path.exists(os.path.join("local-llm", model_id)):
            os.makedirs(os.path.join("local-llm", model_id))
            print("Created directory for LLM")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            model.save_pretrained(os.path.join("local-llm", model_id))
            tokenizer.save_pretrained(os.path.join("local-llm", model_id))
            print("Downloaded LLM")
        else:
            print("LLM already downloaded")

    def load_llm(self) -> BaseLanguageModel:
        model_id = self.get_model_id

        model = AutoModelForSeq2SeqLM.from_pretrained(
            os.path.join("local-llm", model_id)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join("local-llm", model_id)
        )

        pipe = pipeline(
            task="text2text-generation",
            model=model,
            tokenizer=self.tokenizer,
            model_kwargs=self.get_model_arguments,
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

    def load_fake_llm(self):
        self.llm = FakeListLLM()
