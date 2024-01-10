import openai
import pandas as pd
import tiktoken
from scipy import spatial
import typing
import streamlit as st

class OpenAIInquirer:
    def __init__(self, text, df, embedding_model='text-embedding-ada-002', gpt_model= "gpt-4", api_key = None) -> None:
        self.text = text
        self.df = df
        self.embedding_model = embedding_model
        self.gpt_model = gpt_model
        openai.api_key = api_key  

    def _retrieve_ranked_strings(self, max_results: int = 100) -> typing.Tuple[typing.List[str], typing.List[float]]:
        """Internal method to fetch strings ranked by relatedness."""
        query_embedding_response = openai.Embedding.create(model=self.embedding_model, input=self.text)
        print(query_embedding_response)
        query_embedding = query_embedding_response["data"][0]["embedding"]

        def comparison_fn(a, b): return 1 - spatial.distance.cosine(a, b)

        relatedness_scores = [
            (row["text"], comparison_fn(query_embedding, row["embedding"]))
            for _, row in self.df.iterrows()
        ]
        relatedness_scores.sort(key=lambda x: x[1], reverse=True)

        sorted_texts, scores = zip(*relatedness_scores)
        return sorted_texts[:max_results], scores[:max_results]

    def _num_tokens(self, text: str) -> int:
        """Internal method to get the number of tokens."""
        encoding = tiktoken.encoding_for_model(self.gpt_model)
        return len(encoding.encode(text))

    def _construct_query_message(self, token_budget: int) -> str:
        """Builds the query message with context from ranked strings."""
        strings, _ = self._retrieve_ranked_strings()
        introduction = 'Use the information from the Petrel Manual to respond. If no information is found, state "No information available."'
        question = f"\n\nQuestion: {self.text}"
        message = introduction

        for string in strings:
            next_section = f'\n\nSection:\n"""\n{string}\n"""'
            if self._num_tokens(message + next_section + question) > token_budget:
                break
            message += next_section

        return message + question

    def inquire(self, token_budget: int = 4000, print_message: bool = False) -> str:
        """Main method to execute the query and get a response."""
        message = self._construct_query_message(token_budget=token_budget)

        if print_message:
            print(message)

        conversation = [
            {"role": "system", "content": "You provide information on the Petrel Software."},
            {"role": "user", "content": message},
        ]

        response = openai.ChatCompletion.create(model=self.gpt_model, messages=conversation, temperature=0)
        return response["choices"][0]["message"]["content"]



