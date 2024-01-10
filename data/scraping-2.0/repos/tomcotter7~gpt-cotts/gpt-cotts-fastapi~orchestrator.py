"""Orchestrator module."""

from dotenv import load_dotenv
from openai import OpenAI

from database.pinecone import query_pinecone
from utils import load_config

load_dotenv()

class Orchestrator:
    """Orchestrator class - deals with the interaction between the user and the model.

    Attributes:
        query_calls (int): Number of times the model has been queried.
        prompts (dict[str, str]): Dictionary containing the prompts for the model.
        gen_model (str): Name of the model to use for generation.
        embedding_model (str): Name of the model to use for embedding.
        context (list[dict[str, str]]): List of dictionaries containing the context of the conversation.
        client (OpenAI): OpenAI client.
    """

    def __init__(self):  # noqa: D107
        self.query_calls = 0
        config = load_config()
        self.prompts = config["prompts"]
        self.gen_model = config["gen_model"]
        self.embedding_model = config["embedding_model"]
        self.context = [{"role": "system", "content": self.prompts["regular"]}]
        self.client = OpenAI()

    def clear_context(self) -> None:
        """Clears the context and resets the query calls."""
        self.query_calls = 0
        self.context = [{"role": "system", "content": self.prompts["regular"]}]

    def reduce_context_size(self) -> None:
        """Reduces the context size to 3 pairs of messages."""
        self.query_calls = 3
        self.context = self.context[-6:]

    def build_rag_input_prompt(self, input_query: str, details: dict) -> str:
        """Builds the input prompt for RAG.

        Args:
            input_query: The query to be used.
            details: dictionary containing the index and namespace to query.

        Returns:
            The query to be used for RAG, with the context prepended.
        """
        vector_db_result = query_pinecone(
            input_query, details["index"], self.embedding_model, details["namespace"]
        )
        chunks = []
        for chunk in vector_db_result["matches"]:
            chunks.append(chunk["metadata"]["doc"])
        chunks = "\n".join(chunks)
        return f"Potential Context: {chunks} ### Question: {input_query}"

    def query(self, input_query: str, use_rag: bool, details: dict = {}) -> str:
        """Queries the model and returns the response.

        Args:
            input_query: The query to be used.
            use_rag: Whether to use RAG or not.
            details: dictionary containing the index and namespace to query.
                Only used if use_rag is True.

        Returns:
            The response from the model.
        """
        if self.query_calls > 3:
            self.reduce_context_size()

        if use_rag:
            self.context[0] = {"role": "system", "content": self.prompts["rag"]}
            input_query = self.build_rag_input_prompt(input_query, details)
        else:
            self.context[0] = {"role": "system", "content": self.prompts["regular"]}

        self.context.append({"role": "user", "content": input_query})
        self.query_calls += 1
        stream = self.client.chat.completions.create(
            messages=self.context,
            model=self.gen_model,
            stream=True,
        )

        self.context.append({"role": "assistant", "content": ""})

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                self.context[-1]["content"] += chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content
