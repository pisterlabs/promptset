from ast_processor import ASTProcessor
from transformers import GPT2TokenizerFast
from dotenv import load_dotenv
import openai
import os
import time

class Embeddings:
    def __init__(self, workspace_path: str, language: str):
        # Set the workspace path and language
        self.workspace_path = workspace_path

        # Initialize an ASTProcessor for the given language
        self.ast_processor = ASTProcessor(language)

        # Set the models to use for document and query embeddings
        self.DOC_EMBEDDINGS_MODEL = 'text-embedding-ada-002'
        self.QUERY_EMBEDDINGS_MODEL = 'text-embedding-ada-002'

        # Set the separator for tokenization
        self.SEPARATOR = "\n* "

        # Initialize the tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # Compute the length of the separator token sequence
        self.separator_len = len(self.tokenizer.tokenize(self.SEPARATOR))

        # Get the OpenAI API key from environment variables
        openai.api_key = os.getenv("OPENAI_API_KEY", "")

    # ... Rest of the Embeddings class methods