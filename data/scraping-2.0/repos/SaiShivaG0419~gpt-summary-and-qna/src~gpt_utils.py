"""This file to define basic functionalities using Open AI's GPT models. """

import os
import json
import tiktoken  # Importing tiktoken library to calculate the number of tokens
from openai import OpenAI  # Importing Open AI library
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# Load the config.json file
with open(f"{project_root}/config/config.json", "r") as config_file:
    config = json.load(config_file)

# openai.api_key = os.environ["OPENAI_API_KEY"]  # Reading Open AI API Key from environment file
default_model = config["DEFAULT_MODEL"]  # Default gpt model for use - gpt-3.5-turbo
large_context_model = config[
    "LARGE_CONTEXT_MODEL"
]  # Large context gpt model for large amount of tokens - gpt-3.5-turbo-16k


class GPT_UTILS:
    """A class to define various utilities for GPT usage"""

    def __init__(self, api_key) -> None:
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.default_model = default_model
        self.large_context_model = large_context_model
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.langchain_llm = ChatOpenAI(
            openai_api_key=self.api_key, model=self.default_model, temperature=0.5, max_tokens=512
        )
        

    def validate_key(self) -> bool:
        """A function to validate the Open AI API Key"""

        #openai.api_key = self.api_key
        try:
            response = self.client.chat.completions.create(
                model=self.default_model,  # loading default gpt model
                messages=[
                    {"role": "system", "content": "Test Prompt"},
                    {"role": "user", "content": "Hello"},
                ],  # A Test prompt to check if we can get response from Open AI
                max_tokens=5,  # Limit maximum output tokens to 5 to control the cost on each page reload
            )

            if response:
                return True
        except Exception as error:
            print(f"Invalid Key: {error}")  # Terminal Error message for debugging
            return False

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""

        try:
            encoding = tiktoken.encoding_for_model(
                self.default_model
            )  # Loading the correct encoding for default model
            num_tokens = len(
                encoding.encode(string)
            )  # Calculating the length of the tokens
            return num_tokens
        except KeyError:
            print("Warning: model not found.")  # Terminal Error message for debugging
            return None

    def select_model(self, messages, max_tokens):
        """A function to decide the model choice between regular or large context."""

        num_tokens = self.num_tokens_from_string(
            str(messages)
        )  # Get number of prompt tokens

        total_tokens = num_tokens + max_tokens

        if total_tokens is not None and total_tokens < 3750:
            model = (
                self.default_model
            )  # Select default model if prompt tokens are less than 3500
        else:
            model = (
                self.large_context_model
            )  # Select large context model if prompt tokens are more than 3500

        return model

    def get_completion_from_messages(
        self, messages, functions=[], temperature=0.5, max_tokens=1750
    ):
        """A function to get completion from provided messages using GPT models."""

        #openai.api_key = self.api_key
        if len(functions) > 0:
            response = self.client.chat.completions.create(
                model=self.select_model(messages=messages, max_tokens=max_tokens),
                messages=messages,
                tools = [
                    {
                        "type": "function",
                        "function": functions[0]
                    }
                ],
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.select_model(messages=messages, max_tokens=max_tokens),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return response

    def retrieval_qa(self, query, prompt, db, return_source_documents: bool = True):
        """A function to use retrivers from vectorstores and generate completions with GPT models."""

        #openai.api_key = self.api_key
        try:
            retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 6})
            retriever_qa_chain = RetrievalQA.from_chain_type(
                llm=self.langchain_llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=return_source_documents,
                chain_type_kwargs={"prompt": prompt},
            )
            result = retriever_qa_chain({"query": query})

            return result
        except Exception as e:
            print(f"Error retrieving response: {e}")
            return None
