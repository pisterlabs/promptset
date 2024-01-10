from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings


class LangchainOps:
    @staticmethod
    def get_embedder(
        backend: str,
        api_key: str,
        model_name: str = None,
        embeddings_type: str = None
    ) -> Embeddings:
        """Return an embedding object based on the specified backend.

        Args:
            backend: A string specifying the backend to use.
            api_key: A string containing the API key to use for authentication.
            model_name: A string specifying the name of the model to use.
            embeddings_type: A string specifying the type of embeddings to use.

        Returns:
            An embedding object.
        """
        match backend:
            case "openai":
                from langchain.embeddings import OpenAIEmbeddings

                embedder = OpenAIEmbeddings(
                    openai_api_key=api_key,
                    model=(model_name or "text-embedding-ada-002")
                )
                embedder.max_retries = 0
            case "aleph":
                from langchain.embeddings import (
                    AlephAlphaAsymmetricSemanticEmbedding,
                    AlephAlphaSymmetricSemanticEmbedding,
                )

                if embeddings_type == "asymmetric":
                    embedder = AlephAlphaAsymmetricSemanticEmbedding(
                        aleph_alpha_api_key=api_key
                    )
                elif embeddings_type == "symmetric":
                    embedder = AlephAlphaSymmetricSemanticEmbedding(
                        aleph_alpha_api_key=api_key
                    )
            case "spacy":
                from langchain.embeddings import SpacyEmbeddings
                embedder = SpacyEmbeddings()
            case "jina":
                from langchain.embeddings import JinaEmbeddings
                embedder = JinaEmbeddings(
                    jina_auth_token=api_key,
                    model_name=model_name,
                )
            case "hugging_face":
                from langchain.embeddings import HuggingFaceEmbeddings
                embedder = HuggingFaceEmbeddings(model_name=model_name)
            case "cohere":
                from langchain.embeddings import CohereEmbeddings
                embedder = CohereEmbeddings(cohere_api_key=api_key)
            case "dashscope":
                from langchain.embeddings import DashScopeEmbeddings
                embedder = DashScopeEmbeddings(
                    dashscope_api_key=api_key,
                    model=model_name
                )

        return embedder

    @staticmethod
    def get_chat_with_backend(
        backend: str = "openai",
        api_key: str = None,
        temperature: float = 0.5
    ) -> BaseChatModel:
        """Get a chatbot based on the backend.

        Following backend are supported:
            - Open AI (requires API key)
            - Aleph Alpha (requires API key)
            - Jina (requires API key)

        Args:
            backend (str, optional): Chat backend. Defaults to "openai".
            api_key (str, optional): API key. Defaults to None.
            temperature (float, optional): Temperature. Defaults to 0.5.

        Returns:
            Chat: ~langchain.chat_models.base.BaseChatModel object

        """
        if backend == "openai":
            from langchain.chat_models import ChatOpenAI
            chat = ChatOpenAI(openai_api_key=api_key, temperature=temperature)
        elif backend == "aleph":
            from langchain.chat_models import ChatAnthropic
            chat = ChatAnthropic()
        elif backend == "jina":
            # Requires langchain==0.0.228
            from langchain.chat_models import JinaChat
            chat = JinaChat(jina_auth_token=api_key, temperature=temperature)

        return chat


    def __init__(self) -> None:
        self.__chat_model = None

    def attach_chat_model(
        self, backend: str, api_key: str, temperature: float = 0.5
    ) -> None:
        self.__chat_model = LangchainOps.get_chat_with_backend(
            backend=backend,
            api_key=api_key,
            temperature=temperature
        )

    def attach_embedder(
        self,
        backend: str,
        api_key: str,
        model_name: str = None,
        embeddings_type: str = None
    ) -> None:
        self.__embedder = LangchainOps.get_embedder(
            backend=backend,
            api_key=api_key,
            model_name=model_name,
            embeddings_type=embeddings_type
        )

    @property
    def chat_model(self) -> BaseChatModel:
        return self.__chat_model

    @property
    def embedder(self) -> Embeddings:
        return self.__embedder
