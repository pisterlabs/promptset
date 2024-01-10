import langchain.chat_models
import langchain.embeddings


class UrlslabEmbeddingModel:
    def __init__(self,
                 embedding_model_class: str,
                 embedding_model_name: str):
        # Ensure the class name is part of the langchain module
        if embedding_model_class in langchain.embeddings.__all__:
            # Get the class object from the module based on the string
            embedding_class = getattr(langchain.embeddings, embedding_model_class)

            # Create an instance of the class
            self.langchain_model = embedding_class()
            self.embedding_model_class = embedding_model_class
            self.embedding_model_name = embedding_model_name
            self.dimension_size = None
        else:
            raise ValueError(f"Embedding model {embedding_model_class} is not supported")

    @classmethod
    def default(cls):
        return cls(
            embedding_model_class="FastEmbedEmbeddings",
            embedding_model_name="BAAI/bge-small-en",
        )

    def to_dict(self):
        return {
            "embedding_model_class": self.embedding_model_class,
            "embedding_model_name": self.embedding_model_name,
        }

    async def aembedding_dimensions(self):
        if self.dimension_size is None:
            embedding = await self.langchain_model.aembed_query("example text to get embedding dims")
            self.dimension_size = len(embedding)
        return self.dimension_size

    def embedding_dimensions(self):
        if self.dimension_size is None:
            embedding = self.langchain_model.embed_query("example text to get embedding dims")
            self.dimension_size = len(embedding)
        return self.dimension_size

    async def aembed_query(self, query: str):
        return await self.langchain_model.aembed_query(query)

    async def aembed_documents(self, documents: list[str]):
        return await self.langchain_model.aembed_documents(documents)


class UrlslabChatModel:
    def __init__(self,
                 chat_model_class: str,
                 chat_model_name: str):
        # Ensure the class name is part of the langchain module
        if chat_model_class in langchain.chat_models.__all__:
            # Get the class object from the module based on the string
            chat_class = getattr(langchain.chat_models, chat_model_class)

            # Create an instance of the class
            self.langchain_model = chat_class()
            self.chat_model_class = chat_model_class
            self.chat_model_name = chat_model_name
        else:
            raise ValueError(f"Chat model {chat_model_class} is not supported")

    @classmethod
    def default(cls):
        return cls(
            chat_model_class="ChatOpenAI",
            chat_model_name="gpt-3.5-turbo"
        )

    def to_dict(self):
        return {
            "chat_model_class": self.chat_model_class,
            "chat_model_name": self.chat_model_name
        }

