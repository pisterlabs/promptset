from typing import Any

import tiktoken
from fastapi import Depends
from fastapi_pagination import Page, Params
from loguru import logger
from openai import OpenAI

from api.infra.db.model.file import FileChunk
from api.infra.db.repository.file import FileChunkRepository, get_file_chunk_repository


class FileChunkService:
    def __init__(
        self,
        file_chunk_repository: FileChunkRepository,
        openai: OpenAI = OpenAI(),
    ):
        self.file_chunk_repository = file_chunk_repository
        self.openai = openai

    def create_file_chunks_embedding(
        self,
        file_id: int,
        file_text_content: str,
    ) -> None:
        """
        Creates chunks embeddings from a file text content
        :param file_id: The file id
        :param file_text_content: The file text content
        """
        chunks = []
        if self.num_tokens_from_string(file_text_content) <= 512:
            chunks.append(file_text_content)
        else:
            chunks = self.split_text_into_chunks(file_text_content)
        estimated_cost = self.calculate_embedding_cost(file_text_content)
        logger.info(
            f"Embedding file {file_id}, Estimated cost for embedding: {estimated_cost} USD",
        )
        for chunk in chunks:
            embedding = self.create_embedding(chunk)
            file_chunk = FileChunk(
                file_id=file_id,
                chunk_text=chunk,
                embedding_vector=embedding,
            )
            self.file_chunk_repository.create(file_chunk)
        logger.info(f"Finished embedding file {file_id}")

    def create_embedding(self, text: str) -> list[float]:
        """
        Creates an embedding for a text
        :param text: The text to be embedded
        :return: The embedding float list
        """
        response = self.openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text.replace("\n", " "),
        )
        return response.data[0].embedding

    def find_similar_file_chunks(
        self,
        question_embedding: list[float],
        params: Params = Params(),
    ) -> Page[Any]:
        """
        Finds similar top k files to a question embedding using file chunks
        :param params:
        :param question_embedding: The question embedding
        :return: The similar file chunks
        """
        file_chunks = self.file_chunk_repository.find_similar_file_chunks(
            question_embedding,
            params,
        )

        return file_chunks

    @classmethod
    def split_text_into_chunks(cls, text: str) -> list[str]:
        """
        Splits a text into chunks
        :param text: The text to be split
        :return: The chunks
        """
        start = 0
        ideal_token_size = 512
        ideal_size = int(ideal_token_size // (4 / 3))
        end = ideal_size
        words = text.split()
        # remove empty spaces
        words = [x for x in words if x != " "]
        total_words = len(words)
        chunks = []

        nb_chunks = (
            total_words // ideal_size
            if total_words % ideal_size == 0
            else total_words // ideal_size + 1
        )
        for i in range(nb_chunks):
            chunk = words[start:end]
            chunk = " ".join(chunk)
            nb_chunk_token = cls.num_tokens_from_string(chunk)
            if nb_chunk_token > 0:
                chunks.append(chunk)
            start += ideal_size
            end += ideal_size
        return chunks

    @classmethod
    def num_tokens_from_string(cls, string: str, encoding_name="cl100k_base") -> int:
        if not string:
            return 0
        # Returns the number of tokens in a text string
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    @classmethod
    def get_file_words_length(cls, file_text_content: str) -> int:
        """
        Returns the number of words in a file
        :param file_text_content: The file text content
        :return: The number of words
        """
        return len(file_text_content.split())

    @classmethod
    def calculate_embedding_cost(cls, file_text_content: str) -> float:
        """
        Calculates the cost of embedding a file
        :param file_text_content: The file text content
        :return: The cost
        """
        num_tokens = cls.num_tokens_from_string(file_text_content)
        cost = num_tokens / 1000 * 0.0001
        return round(cost, 4)


def get_file_chunk_service(
    file_chunk_repository: FileChunkRepository = Depends(get_file_chunk_repository),
) -> FileChunkService:
    return FileChunkService(file_chunk_repository)
