"""
# Document Chatbot
A chatbot that uses a knowledge base to answer questions. The knowledge base is a vector store that contains the documents. The embeddings model is used to embed the documents and the prompt. The model is used to generate the response.
"""

import os
import tempfile
import uuid
from datetime import datetime
from time import perf_counter_ns
from typing import Dict, List, Literal, Tuple

from langchain.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain.document_loaders.base import BaseLoader
from typing_extensions import override

from softtek_llm.cache import Cache
from softtek_llm.chatbots.chatbot import Chatbot
from softtek_llm.embeddings import EmbeddingsModel
from softtek_llm.exceptions import InvalidPrompt, KnowledgeBaseEmpty
from softtek_llm.memory import Memory, WindowMemory
from softtek_llm.models import LLMModel
from softtek_llm.schemas import Filter, Message, Response, Vector
from softtek_llm.utils import strip_accents_and_special_characters
from softtek_llm.vectorStores import VectorStore


class DocumentChatBot(Chatbot):
    """
    # Document Chatbot
    A chatbot that uses a knowledge base to answer questions. The knowledge base is a vector store that contains the documents. The embeddings model is used to embed the documents and the prompt. The model is used to generate the response. Inherits from `Chatbot`.

    ## Attributes
    - `knowledge_base` (VectorStore): The vector store that contains the documents.
    - `embeddings_model` (EmbeddingsModel): The embeddings model to use for embedding the documents and the prompt.
    - `knowledge_base_namespace` (str): Namespace used by the knowledge base in the vector store.

    ## Methods
    - `add_document`: Adds a document to the knowledge base.
    - `delete_document`: Deletes a document from the knowledge base.
    - `chat`: Chatbot function that returns a response given a prompt. If a memory and/or cache are available, it considers previously stored conversations. Filters are applied to the prompt before processing to ensure it is valid.
    """

    __extension_mapper = {
        "pdf": PyPDFLoader,
        "docx": Docx2txtLoader,
        "txt": TextLoader,
        "doc": Docx2txtLoader,
        "csv": CSVLoader,
    }

    @override
    def __init__(
        self,
        model: LLMModel,
        knowledge_base: VectorStore,
        embeddings_model: EmbeddingsModel,
        description: str = "You are a helpful research assistant. You have access to documents and always respond using the most relevant information.",
        memory: Memory = WindowMemory(window_size=10),
        non_valid_response: str | None = None,
        filters: List[Filter] | None = None,
        cache: Cache | None = None,
        cache_probability: float = 0.5,
        verbose: bool = False,
        knowledge_base_namespace: str | None = None,
    ):
        """Chatbot that uses a knowledge base to answer questions. The knowledge base is a vector store that contains the documents. The embeddings model is used to embed the documents and the prompt. The model is used to generate the response.

        Args:
            `model` (LLMModel): The LLM to use for generating the response.
            `knowledge_base` (VectorStore): The vector store that contains the documents.
            `embeddings_model` (EmbeddingsModel): The embeddings model to use for embedding the documents and the prompt.
            `description` (str, optional): Information about the bot. Defaults to "You are a helpful research assistant. You have acess to documents and always respond using the most relevant information.".
            `memory` (Memory, optional): The memory to use. Defaults to WindowMemory(window_size=10).
            `non_valid_response` (str | None, optional): Response given when the prompt does not follow the rules set by the filters. Defaults to None. If `None`, an `InvalidPrompt` exception is raised when the prompt does not follow the rules set by the `filters`.
            `filters` (List[Filter] | None, optional): List of filters used by the chatbot. Defaults to None.
            `cache` (Cache | None, optional): Cache used by the chatbot. Defaults to None.
            `cache_probability` (float, optional): Probability of using the cache. Defaults to 0.5. If 1.0, the cache is always used. If 0.0, the cache is never used.
            `verbose` (bool, optional): Whether to print additional information. Defaults to False.
            `knowledge_base_namespace` (str | None, optional): Namespace used by the knowledge base in the vector store. Defaults to None.
        """
        super().__init__(
            model,
            description,
            memory,
            non_valid_response,
            filters,
            cache,
            cache_probability,
            verbose,
        )

        self.__knowledge_base = knowledge_base
        self.__embeddings_model = embeddings_model
        self.knowledge_base_namespace = knowledge_base_namespace

    @property
    def knowledge_base(self) -> VectorStore:
        """The vector store that contains the documents."""
        return self.__knowledge_base

    @property
    def embeddings_model(self) -> EmbeddingsModel:
        """The embeddings model to use for embedding the documents and the prompt."""
        return self.__embeddings_model

    @property
    def knowledge_base_namespace(self) -> str:
        """Namespace used by the knowledge base in the vector store."""
        return self.__knowledge_base_namespace

    @knowledge_base_namespace.setter
    def knowledge_base_namespace(self, knowledge_base_namespace: str):
        if (
            not isinstance(knowledge_base_namespace, str)
            and knowledge_base_namespace is not None
        ):
            raise TypeError("knowledge_base_namespace must be a string or None")
        self.__knowledge_base_namespace = knowledge_base_namespace

    def __get_document_name_and_file_path(
        self, file: str | bytes, file_type: Literal["pdf", "doc", "docx", "txt"]
    ) -> Tuple[str, str]:
        """Extracts the document name and the file path from the file.

        Args:
            `file` (str | bytes): Either the path to the file or the bytes of the file.
            `file_type` (Literal["pdf", "doc", "docx", "txt"]): The type of the file.

        Raises:
            `ValueError`: If unsupported file_type is provided.
            `FileNotFoundError`: If the file does not exist.
            `ValueError`: If document_name is not provided and file is bytes.
            `TypeError`: If file is not str or bytes.

        Returns:
            `Tuple[str, str]`: The document name and the file path.
        """
        # * Check valid file_type
        if file_type not in self.__extension_mapper.keys():
            raise ValueError(
                f"file_type must be one of {self.__extension_mapper.keys()}"
            )

        # * Read file
        if isinstance(file, str):
            if not os.path.exists(file):
                raise FileNotFoundError(f"{file} does not exists")
            document_name = ".".join(os.path.basename(file).split(".")[:-1])
            file_path = file
        elif isinstance(file, bytes):
            if document_name is None:
                raise ValueError("document_name must be provided, unless file is str")

            temporal_dir = tempfile.gettempdir()
            file_path = os.path.join(temporal_dir, f"{str(uuid.uuid4())}.{file_type}")
            with open(file_path, "wb") as f:
                f.write(file)
        else:
            raise TypeError("file must be str or bytes")

        return document_name, file_path

    def __split_document(
        self, file_type: Literal["pdf", "doc", "docx", "txt"], file_path: str
    ) -> List[str]:
        """Splits the document into chunks.

        Args:
            `file_type` (Literal["pdf", "doc", "docx", "txt"]): The type of the file.
            `file_path` (str): The path to the file.

        Returns:
            `List[str]`: The chunks of the document.
        """
        loader: BaseLoader = self.__extension_mapper[file_type](file_path)
        file_data = loader.load_and_split()
        content = [page.page_content for page in file_data]

        return content

    def __get_vectors(
        self,
        file_type: Literal["pdf", "doc", "docx", "txt"],
        file_path: str,
        document_name: str,
    ) -> List[Vector]:
        """Embeds the document and returns the vectors.

        Args:
            `file_type` (Literal["pdf", "doc", "docx", "txt"]): The type of the file.
            `file_path` (str): The path to the file.
            `document_name` (str): The name of the document.

        Returns:
            `List[Vector]`: The embedded vectors of the document.
        """
        content = self.__split_document(file_type, file_path)
        embedded_file = [self.embeddings_model.embed(page) for page in content]
        vectors = [
            Vector(
                embeddings=page,
                id=f"{strip_accents_and_special_characters(document_name)}_{i}",
                metadata={"source": f"{document_name}.{file_type}", "text": content[i]},
            )
            for i, page in enumerate(embedded_file)
        ]

        return vectors

    def add_document(
        self,
        file: str | bytes,
        file_type: Literal["pdf", "doc", "docx", "txt", "csv"],
        document_name: str | None = None,
    ):
        """Adds a document to the knowledge base.

        Args:
            `file` (str | bytes): Either the path to the file or the bytes of the file.
            `file_type` (Literal["pdf", "doc", "docx", "txt", "csv"]): The type of the file.
            `document_name` (str | None, optional): The name of the document. Defaults to None. If None, the name of the file is used.
        """
        document_name, file_path = self.__get_document_name_and_file_path(
            file, file_type
        )
        vectors = self.__get_vectors(file_type, file_path, document_name)
        self.knowledge_base.add(vectors, namespace=self.knowledge_base_namespace)

    def delete_document(
        self,
        file: str | bytes,
        file_type: Literal["pdf", "doc", "docx", "txt", "csv"],
        document_name: str | None = None,
    ):
        """Deletes a document from the knowledge base.

        Args:
            `file` (str | bytes): Either the path to the file or the bytes of the file.
            `file_type` (Literal["pdf", "doc", "docx", "txt", "csv"]): The type of the file.
            `document_name` (str | None, optional): The name of the document. Defaults to None. If None, the name of the file is used.
        """
        document_name, file_path = self.__get_document_name_and_file_path(
            file, file_type
        )
        vector_count = len(self.__split_document(file_type, file_path))
        self.knowledge_base.delete(
            [
                f"{strip_accents_and_special_characters(document_name)}_{i}"
                for i in range(vector_count)
            ],
            namespace=self.knowledge_base_namespace,
        )

    def __call_model(
        self,
        include_context: bool,
        top_documents: int,
        logging_kwargs: Dict | None = None,
    ) -> Response:
        """This method is used to call the model and returns a Response object.


        Args:
            `include_context` (bool): Whether to include the context in the response.
            `top_documents` (int): The number of documents to consider.
            `logging_kwargs` (Dict | None, optional): Additional keyword arguments to be passed to the logging function. Defaults to None.

        Returns:
            `Response`: The response of the model.
        """
        # * Embed prompt
        last_message = self.memory.get_message(-1)
        embeddings = self.embeddings_model.embed(last_message.content)
        vector = Vector(embeddings=embeddings)

        # * Get similar vectors
        similar_vectors = self.knowledge_base.search(
            vector=vector,
            namespace=self.knowledge_base_namespace,
            top_k=top_documents,
        )

        # * Extract context
        all_sources = [vector.metadata["source"] for vector in similar_vectors]
        if all_sources:
            sources = []
            for source in all_sources:
                if source not in sources:
                    sources.append(source)
        else:
            raise KnowledgeBaseEmpty("The knowledge base is empty")

        context = "\n".join([vector.metadata["text"] for vector in similar_vectors])

        # * Prepare new memory
        messages = self.memory.get_messages()[:-1]
        memory = Memory.from_messages(messages)
        memory.add_message(
            role="user",
            content=f"Considering this context: {context}\nAnswer this: {last_message.content}",
        )

        # * Call model
        response = (
            self.model(self.memory, description=self.description)
            if logging_kwargs is None
            else self.model(
                self.memory, description=self.description, logging_kwargs=logging_kwargs
            )
        )

        # * Update original memory
        self.memory.add_message(**response.message.model_dump())

        # * Update response
        response.additional_kwargs.update({"sources": sources})
        if include_context:
            response.additional_kwargs.update({"context": context})

        return response

    @override
    def chat(
        self,
        prompt: str,
        print_cache_score: bool = False,
        include_context: bool = False,
        top_documents: int = 5,
        cache_kwargs: Dict = {},
        logging_kwargs: Dict | None = None,
    ) -> Response:
        """Chatbot function that returns a response given a prompt. If a memory and/or cache are available, it considers previously stored conversations. Filters are applied to the prompt before processing to ensure it is valid.

        Args:
            `prompt` (str): User's input string text.
            `print_cache_score` (bool, optional): Whether to print the cache score. Defaults to False.
            `include_context` (bool, optional): Whether to include the context in the response. Defaults to False.
            `top_documents` (int, optional): The number of documents to consider. Defaults to 5.
            `cache_kwargs` (Dict, optional): Additional keyword arguments to be passed to the cache. Defaults to {}.
            `logging_kwargs` (`Dict`, optional): additional keyword arguments to be passed to the logging function. **Can only be used with certain models**. Defaults to `None`.

        Raises:
            `InvalidPrompt`: If the prompt does not follow the rules set by the filters and `non_valid_response` is None.

        Returns:
            `Response`: The response given by the chatbot. Whithin the additional_kwargs, the following keys are available: `sources` (always), `context` (if `include_context` is `True`).
        """
        start = perf_counter_ns()
        if self.filters:
            if not self._revise(prompt):
                if self.non_valid_response:
                    return Response(
                        message=Message(role="system", content=self.non_valid_response),
                        created=int(datetime.utcnow().timestamp()),
                        latency=int((perf_counter_ns() - start) / 1e6),
                        from_cache=False,
                        model="reviser",
                    )
                raise InvalidPrompt(
                    "The prompt does not follow the rules set by the filters. If this behavior is not intended, consider modifying the filters. It is recommended to use LLMs for meta prompts."
                )

        self.memory.add_message(role="user", content=prompt)
        if not self.cache:
            last_message = self.__call_model(
                include_context, top_documents, logging_kwargs
            )
        else:
            if self._random_boolean():
                cached_response, cache_score = self.cache.retrieve(
                    prompt=prompt, **cache_kwargs
                )
                if print_cache_score:
                    print(f"Cache score: {cache_score}")
                if cached_response:
                    self.memory.add_message(
                        role=cached_response.message.role,
                        content=cached_response.message.content,
                    )
                    last_message = cached_response
                else:
                    last_message = self.__call_model(
                        include_context, top_documents, logging_kwargs
                    )
                    self.cache.add(prompt=prompt, response=last_message, **cache_kwargs)
            else:
                last_message = self.__call_model(
                    include_context, top_documents, logging_kwargs
                )

        return last_message
