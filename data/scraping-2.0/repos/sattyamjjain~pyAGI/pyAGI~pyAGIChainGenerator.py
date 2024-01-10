from langchain import LLMChain, PromptTemplate, OpenAI
from langchain.chat_models import ChatOpenAI

from pyAGI.prompts import CHAIN_GENERATOR_PROMPT

_UPCOMING_OPENAI_MODELS = [
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-4-32k-0314",
]  # you can join the wait-list for this https://openai.com/waitlist/gpt-4-api

_AVAILABLE_OPENAI_MODELS = [
    "text-davinci-002",
    "text-davinci-003",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
]

CHAT_BASED_OPEN_AI_MODELS = ["gpt-3.5-turbo", "gpt-3.5-turbo-0301"]

TEXT_BASED_OPEN_AI_MODELS = [
    "text-davinci-002",
    "text-davinci-003",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
]


class GeneratePyAGIChain(LLMChain):
    @classmethod
    def create_chain(
        cls, verbose: bool = False, selected_model: str = None
    ) -> LLMChain:
        if selected_model in _UPCOMING_OPENAI_MODELS:
            raise AssertionError(
                f"{selected_model} model will not be supported until its released"
            )
        if selected_model not in _UPCOMING_OPENAI_MODELS + _AVAILABLE_OPENAI_MODELS:
            raise AssertionError(f"UnSupported OpenAI model: {selected_model}")
        if cls.is_chat_model(selected_model):
            llm = ChatOpenAI(model_name=selected_model, temperature=0.3)
        else:
            llm = OpenAI(model_name=selected_model, temperature=0.3)
        chain_instance = cls(
            prompt=PromptTemplate(
                template=CHAIN_GENERATOR_PROMPT,
                input_variables=["objective", "maincontent", "outcome"],
            ),
            llm=llm,
        )
        return chain_instance

    @staticmethod
    def is_chat_model(selected_model: str) -> bool:
        return selected_model in CHAT_BASED_OPEN_AI_MODELS
