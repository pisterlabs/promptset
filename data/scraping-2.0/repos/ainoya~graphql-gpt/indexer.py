# Class to create Chroma index from GraphQL schema 
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from typing import List

from langchain.vectorstores import Chroma
from graphql_schema_splitter import HasuraGraphQLSchemaSplitter


class Indexer:
    def __init__(self, graphql_schema, llm: OpenAI):
        self.llm = llm
        self._graphql_schema = graphql_schema

    def run(self):
        # Register the text into the Chroma index
        # Split the GraphQL schema into chunks of 3000 characters

        text_splitter = HasuraGraphQLSchemaSplitter(chunk_size=3000)
        texts = text_splitter.split_text(self._graphql_schema)

        # Initialize the OpenAI embeddings
        embeddings = OpenAIEmbeddings()
        # Create the Chroma index from the texts and embeddings
        docsearch = Chroma.from_texts(texts, embeddings)

        return docsearch
