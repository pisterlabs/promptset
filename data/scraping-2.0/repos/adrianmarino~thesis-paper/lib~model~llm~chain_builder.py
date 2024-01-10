from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate


class OllamaModelBuilder:
    @staticmethod
    def chat(model='movie_recommender', verbose=False):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]) if verbose else None
        return ChatOllama(model=model, callback_manager=callback_manager)

    @staticmethod
    def default(model='movie_recommender', verbose=False):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]) if verbose else None
        return Ollama(model=model, callback_manager=callback_manager)


class OllamaChatPromptTemplateFactory:
    @staticmethod
    def create(prompt):
        return ChatPromptTemplate.from_messages([
            ('system', prompt),
            ('human', '{request}')
        ])


class OllamaChainBuilder:
    @staticmethod
    def default(model, prompt, verbose=False):
        return OllamaChatPromptTemplateFactory.create(prompt) | \
            OllamaModelBuilder.default(model, verbose)

    @staticmethod
    def chat(model, prompt, verbose=False):
        return OllamaChatPromptTemplateFactory.create(prompt) | \
            OllamaModelBuilder.chat(model, verbose)
