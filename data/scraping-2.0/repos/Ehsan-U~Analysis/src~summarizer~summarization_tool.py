import os
import re
import openai
from src.logger import logging
from src.summarizer.summarizer_prompts import map_prompt, reduce_prompt

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback


openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class Summarizer:
    """
    This is a document summarizer that is responsible summarizing the scrapped text
    """
    def __init__(self, user_settings:dict):
        """
        Initializes the summarizer with user settings.

        Args:
            user_settings (dict): A dictionary containing settings for the ingestor, such as the OpenAI model name and collection name for the vector store.
        """

        self._reduction_max_tokens = 4000
        self.connect_to_llm(user_settings["openAI_model_name"])
        self.define_summary_text_splitter()
        self._initialize_summarizer()

    def connect_to_llm(self, model_name:str):
        """
        Establishes a connection to the language model (LLM).

        Args:
            model_name (str): The name of the OpenAI model to be used.
        """
        logging.info("Establishing connection to open ai models for summarization")
        try: 
            self._llm = ChatOpenAI(model_name= model_name, temperature=0)
            self._embedding_llm = OpenAIEmbeddings(
                            model="text-embedding-ada-002",
                        )
        except Exception as excep:
            logging.error(f"Error establishing connection to open ai models for summarization: {excep}")

    def _create_content_extraction_list(self, company_name: str, keywords: list):
        final_string = ""
        for i, key in enumerate(keywords):
            content = f"{i+1}. Information about {key} for {company_name}: \n"
            final_string += content
        return final_string[:-1]

    def define_summary_text_splitter(self):
        """
        Defines text splitter for when we are splitting to store into the vector database
        """
        self._sum_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                                chunk_size=8000, chunk_overlap=0.5
                            )

    def _initialize_summarizer(self):
        """
        Initializes the summarization process by setting up various chains and reducers for document processing and summarization.
        """
        map_chain = LLMChain(llm=self._llm, prompt=map_prompt)

        # Run chain
        reduce_chain = LLMChain(llm=self._llm, prompt=reduce_prompt)

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        # Combines and iteravely reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=self._reduction_max_tokens,
        )

        # Combining documents by mapping a chain over them, then combining results
        self._map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )

    def _text_preprocessor(self, text):
        """
        Preprocesses the given text by removing HTTP links, empty brackets, and normalizing spaces.

        Args:
            text (str): The text to preprocess.

        Returns:
            str: The cleaned and preprocessed text.
        """
        # Define a regular expression pattern to match HTTP links
        http_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        # Use re.sub to replace HTTP links with an empty string
        cleaned_text = re.sub(http_pattern, '', text)
        # Define a regular expression pattern to match empty square brackets
        empty_brackets_pattern = r'\[\s*\]'
        # Use re.sub to replace empty square brackets with an empty string
        cleaned_text = re.sub(empty_brackets_pattern, '', cleaned_text)
        # Define a regular expression pattern to match square brackets without alphabets
        no_alphabet_brackets_pattern = r'\[[^\w]*\]'
        # Use re.sub to replace square brackets without alphabets with an empty string
        cleaned_text = re.sub(no_alphabet_brackets_pattern, '', cleaned_text)
        # Use a regular expression to replace multiple spaces with a single space
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        # Use regular expression to replace consecutive "\n" with a single "\n"
        cleaned_text = re.sub(r'\n+', '\n', cleaned_text)

        return cleaned_text

    def _summarize(self, text: str, company_name: str, keywords: list):
        """
        Summarizes the given split documents for a specified company.

        Args:
            text: The document to summarize as string.
            company: The company name for which the summarization is being done as string.

        Returns:
            The summarized content.
        """
        information_to_extract = self._create_content_extraction_list(company_name, keywords)
        split_documents = self._sum_text_splitter.create_documents([text])
        for idx in range(len(split_documents)):
            split_documents[idx].metadata['company_name'] = company_name
        return self._map_reduce_chain.run(input_documents=split_documents, company_name = company_name, information_to_extract = information_to_extract)

    def process(self, text, company_name, keywords):
        """
        Preprocesses, summarizes, the given text
        Args:
            text (str): The text content representing the scraped data.
            company: The company associated with the text.

        Returns:
            A tuple containing the summarized content and token consumption information.
        """
        

        text = self._text_preprocessor(text)
        with get_openai_callback() as cb: 
            logging.info("Summarizing the text for vector database")
            summarized_content = self._summarize(text, company_name, keywords)
            
        return summarized_content