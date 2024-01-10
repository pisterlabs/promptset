from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
)
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import openai
import json


class DittoExampleStore:
    def __init__(self):
        # example template will be used to match query / example pairs.
        self.example_template = PromptTemplate(
            input_variables=["user", "ditto"],
            template="User: {user}\nDitto: {ditto}",
        )
        self.construct()

    def construct(self):
        self.create_query_example_pairs()
        self.create_example_store()

    def create_query_example_pairs(self):
        """creates query / example pairs and saves to self.example_pairs"""

        self.example_pairs = [
            {
                "user": "What are gas prices looking like today?",
                "ditto": "<GOOGLE_SEARCH> gas prices today",
            },
            {
                "user": "What is the weather like in New York?",
                "ditto": "<GOOGLE_SEARCH> weather in New York",
            },
            {
                "user": "Can you look up topic X or topic Y for me?",
                "ditto": "<GOOGLE_SEARCH> topic X or topic Y",
            },
            {
                "user": "What is the cheapest flight to New York from San Francisco?",
                "ditto": "<GOOGLE_SEARCH> cheapest flight from San Francisco to New York",
            },
            {
                "user": "Latest news on topic X or topic Y?",
                "ditto": "<GOOGLE_SEARCH> latest news on topic X",
            },
            {
                "user": "What's the weather in Atlanta, GA?",
                "ditto": "<GOOGLE_SEARCH> weather in Atlanta, GA",
            },
            {
                "user": "What is the current population of Tokyo, Japan?",
                "ditto": "<GOOGLE_SEARCH> population of Recife, Brazil",
            },
            {
                "user": "Can you look up the movie Back to the Future for me?",
                "ditto": "<GOOGLE_SEARCH> movie Back to the Future",
            },
            {
                "user": "Google search the movie The Matrix",
                "ditto": "<GOOGLE_SEARCH> movie The Matrix",
            },
            {
                "user": "What is the forecast for the next 5 days in Miami, FL?",
                "ditto": "<GOOGLE_SEARCH> forecast for the next 5 days in Miami, FL",
            },
            {
                "user": "can you look up the movie Fear and Loathing in Las Vegas and tell me a summary of the description",
                "ditto": "<GOOGLE_SEARCH> movie Fear and Loathing in Las Vegas",
            },
            {
                "user": "Can you use google to search for the latest news involving aquaponics?",
                "ditto": "<GOOGLE_SEARCH> latest news involving aquaponics",
            },
            {
                "user": "Can you look up the weather in Golden, CO?",
                "ditto": "<GOOGLE_SEARCH> weather in Golden, CO",
            },
            {
                "user": "What is the current time in New York?",
                "ditto": "<GOOGLE_SEARCH> current time in New York",
            },
            {
                "user": "Can you google search topic X or topic Y for me?",
                "ditto": "<GOOGLE_SEARCH> topic X or topic Y",
            },
            {
                "user": "Google search the latest news on topic X or topic Y for me?",
                "ditto": "<GOOGLE_SEARCH> latest news on topic X",
            },
            {
                "user": "Can you try looking up the weather in Maimi FL using Google?",
                "ditto": "<GOOGLE_SEARCH> weather in Miami, FL",
            },
            {
                "user": "Write me a python script that says hello world.",
                "ditto": "<PYTHON_AGENT> write a script that says hello world",
            },
            {
                "user": "Can you code up a quick python script that is the game pong?",
                "ditto": "<PYTHON_AGENT> write a script that is the game pong",
            },
            {
                "user": "can you make a simple timer app in python? Code:",
                "ditto": "<PYTHON_AGENT> make a simple timer app in python",
            },
            {
                "user": "Write me a Python script that generates the Fibonacci sequence up to a specified number.",
                "ditto": "<PYTHON_AGENT> write a script that generates the Fibonacci sequence up to a specified number.",
            },
            {
                "user": "I need a Python script that creates a basic machine learning model using scikit-learn to predict a target variable from a dataset.",
                "ditto": "<PYTHON_AGENT> provide a script that creates a basic machine learning model using scikit-learn to predict a target variable from a dataset.",
            },
            {
                "user": "openscad for a simple sphere.",
                "ditto": "<OPENSCAD_AGENT> openscad for a simple sphere.",
            },
            {
                "user": "Can you make me a computer mouse in OpenSCAD.",
                "ditto": "<OPENSCAD_AGENT> make me a computer mouse in OpenSCAD.",
            },
            {
                "user": "Can you design a cube in OpenSCAD?",
                "ditto": "<OPENSCAD_AGENT> design a cube in OpenSCAD?",
            },
        ]

    def create_example_store(self):
        embeddings = OpenAIEmbeddings()
        self.example_store = MaxMarginalRelevanceExampleSelector.from_examples(
            # This is the list of examples available to select from.
            self.example_pairs,
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            embeddings,
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # This is the number of examples to produce.
            k=5,
        )
        self.mmr_prompt = FewShotPromptTemplate(
            # We provide an ExampleSelector instead of examples.
            example_selector=self.example_store,
            example_prompt=self.example_template,
            prefix="Below are some examples of how how to use the tools:",
            suffix=" {user}",
            input_variables=["user"],
        )

    def get_example(self, query: str):
        example = str(self.mmr_prompt.format(user=query)).replace(query, "")
        return example


if __name__ == "__main__":
    ditto_example_store = DittoExampleStore()
