import logging
import json

from openai import OpenAI
from pydantic import BaseModel
import postcode.types.openai as openai_types
from postcode.databases.chroma.chromadb_collection_manager import (
    ChromaCollectionManager,
)
from postcode.ai_services.librarians.prompts.prompt_creator import (
    ChromaLibrarianPromptCreator,
)
from postcode.ai_services.librarians.prompts.chroma_librarian_prompts import (
    DEFAULT_CHROMA_LIBRARIAN_PROMPT,
    DEFAULT_CHROMA_LIBRARIAN_SYSTEM_PROMPT,
)
import postcode.types.chroma as chroma_types

# TOOLS: list[dict[str, Any]] = [
#     {
#         "type": "function",
#         "function": {
#             "name": "query_chroma",
#             "description": "Get the results from the chromadb vector database using a list of queries.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "queries": {
#                         "type": "list[str]",
#                         "description": "List of queries to use to get the results from the chromadb vector database.",
#                     },
#                     "n_results": {
#                         "type": "int",
#                         "description": "Number of results to return, default is 10.",
#                     },
#                 },
#                 "required": ["queries"],
#             },
#         },
#     }
# ]


class OpenAIResponseContent(BaseModel):
    """
    Pydantic model representing the content structure of an OpenAI response.

    OpenAI is set to respond with a JSON object, so this model is used to parse the response.

    Attributes:
        - query_list (list[str]): List of queries in the OpenAI response.
    """

    query_list: list[str]


class ChromaLibrarian:
    def __init__(
        self,
        collection_manager: ChromaCollectionManager,
        model: str = "gpt-3.5-turbo-1106",
    ) -> None:
        """
        Represents a librarian for interacting with the Chroma database using OpenAI.

        Args:
            - collection_manager (ChromaCollectionManager): The manager for Chroma collections.
            - model (str, optional): The OpenAI model to use. Defaults to "gpt-3.5-turbo-1106".

        Methods:
            - query_chroma(user_question):
                Queries the Chroma database using the provided user question.

            - _query_collection(queries, n_results=3):
                Queries the Chroma collection manager with a list of queries.

            - _get_chroma_queries(user_question, queries_count=3, retries=3):
                Generates Chroma queries based on the user question.

        Attributes:
            - collection_manager (ChromaCollectionManager): The Chroma collection manager.
            - model (str): The OpenAI model being used.
            - client: The OpenAI API client.

        Examples:
            ```python
            chroma_librarian = ChromaLibrarian(chroma_collection_manager)
            chroma_librarian.query_chroma("Which models are inherited by others?")
            ```
        """

        self.collection_manager: ChromaCollectionManager = collection_manager
        self.model: str = model
        self.client = OpenAI()

    def query_chroma(self, user_question: str) -> chroma_types.QueryResult | None:
        """
        Queries the Chroma database using the provided user question.

        Args:
            - user_question (str): The user's question.

        Returns:
            - chroma_types.QueryResult | None: The result of the Chroma query, or None if unsuccessful.
        """

        queries: list[str] | None = self._get_chroma_queries(user_question)
        if not queries:
            return None

        print(queries)

        return self._query_collection(queries)

    def _query_collection(
        self,
        queries: list[str],
        n_results: int = 3,
    ) -> chroma_types.QueryResult | None:
        """
        Queries the Chroma collection manager with a list of queries.

        Args:
            - queries (list[str]): List of queries to use in the Chroma collection manager.
            - n_results (int, optional): Number of results to return. Defaults to 3.

        Returns:
            - chroma_types.QueryResult | None: The result of the Chroma query, or None if unsuccessful.
        """

        return self.collection_manager.query_collection(
            queries,
            n_results=n_results,
            include_in_result=["metadatas", "documents"],
        )

    def _get_chroma_queries(
        self, user_question: str, queries_count: int = 3, retries: int = 3
    ) -> list[str] | None:
        """
        Generates Chroma queries based on the user question.

        Args:
            - user_question (str): The user's question.
            - queries_count (int, optional): Number of queries to generate. Defaults to 3.
            - retries (int, optional): Number of retries in case of failure. Defaults to 3.

        Returns:
            - list[str] | None: The generated list of Chroma queries, or None if unsuccessful.
        """

        while retries > 0:
            retries -= 1

            prompt: str = ChromaLibrarianPromptCreator.create_prompt(
                user_question,
                prompt_template=DEFAULT_CHROMA_LIBRARIAN_PROMPT,
                queries_count=queries_count,
            )

            try:
                completion: openai_types.ChatCompletion = (
                    self.client.chat.completions.create(
                        model=self.model,
                        response_format={"type": "json_object"},
                        messages=[
                            {
                                "role": "system",
                                "content": DEFAULT_CHROMA_LIBRARIAN_SYSTEM_PROMPT,
                            },
                            {"role": "user", "content": prompt},
                        ],
                    )
                )
                content: str | None = completion.choices[0].message.content
                if not content:
                    continue

                content_json = json.loads(content)
                content_model = OpenAIResponseContent(
                    query_list=content_json["query_list"]
                )
                content_model.query_list.append(user_question)
                queries_count += 1

                if content:
                    queries: list[str] = content_model.query_list
                    if queries and len(queries) == queries_count:
                        return queries

            except Exception as e:
                logging.error(f"An error occurred: {e}")

        return None
