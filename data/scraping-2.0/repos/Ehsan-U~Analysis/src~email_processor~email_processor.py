from dotenv import load_dotenv
import os
import openai
load_dotenv()
from langchain.chat_models import ChatOpenAI


from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.callbacks import get_openai_callback

from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.schema import SystemMessage

from src.email_processor.prompts import first_prompt, second_prompt, third_prompt
from langchain.output_parsers import CommaSeparatedListOutputParser

from src.logger import logging

import re

class EmailProcesssor:
    """
    This class processes emails using a chain of language models to identify and categorize patterns in the emails. It connects to a language model and defines a processing chain for analyzing email content.
    """
    def __init__(self, user_settings:dict):
        self.connect_to_llm(user_settings["openAI_model_name"])
        self._initialize_identification_chain()

    def connect_to_llm(self, model_name:str):
        """
        establish connection to the llm
        """

        logging.info("Establishing connection to open ai models for email processing")
        try:
            self._llm = ChatOpenAI(model_name=model_name, temperature=0)

            messages = [
                        SystemMessage(
                                content="You are a helpful assistant that specialize in identifying patterns in email addresses of a company. You are proficient in recognizing names in hindi, chinese, english, french, spanish, italian, german and more. You are able to expertly filter email addresses that belong to individuals in the company and identify patterns in the email address structure specifically in the local-part of the user email addresses."
                        )
                        ]
            self._llm(messages)
        except Exception as excep:
            logging.error(f"Error establishing connection to llm for email processing: {excep}")

    def _initialize_identification_chain(self):
        """
        Defines the identification chain, which is a sequence of LLMChains, each with a specific prompt and output key. This chain is responsible for processing emails and identifying patterns.

        The chain consists of three steps, each using a different prompt to guide the language model in processing and categorizing email content.
        """

        logging.info("Initializing email pattern recognition chains")
        try:
            chain_one = LLMChain(llm=self._llm, prompt=first_prompt, output_key="emails")
            chain_two = LLMChain(llm=self._llm, prompt=second_prompt, output_key="patterns")
            
            output_parser = CommaSeparatedListOutputParser()        
            chain_three = LLMChain(llm=self._llm, prompt=third_prompt, output_key = "final_result", output_parser=output_parser)
            

            self.main_chain = SimpleSequentialChain(chains=[chain_one, chain_two, chain_three],
                                                verbose=True
                                                )
        except Exception as excep:
            logging.error(f"Error initializing chains: {excep}")

    def _post_process_text(self, patterns: str):
        """
        Post-processes the identified patterns to normalize and simplify the representation.

        Args:
            patterns (str): The string of patterns identified by the language model.

        Returns:
            str: A post-processed string where certain placeholders and formats have been standardized.
        """
        #return none if no patterns were found
        if patterns.strip().lower() == "NONE".lower():
            return None

        patterns = patterns.strip().split("@")[0]
        patterns = patterns.replace("[First Name]", "f")
        patterns = patterns.replace("[Last Name]", "l")
        patterns = patterns.replace("[First Name Initial]", "f1")
        patterns = patterns.replace("[Last Name Initial]", "l1")

        return patterns

    def process_emails(self, emails: list):
        """
        Processes a list of emails to identify patterns using the defined identification chain.

        Args:
            emails (list): A list of email strings to be processed.

        Returns:
            str or None: The identified patterns in a post-processed format, or None if no patterns are identified.
        """
        logging.info("Finding patterns in emails")
        try:
            if len(emails) == 0:
                return None
            
            #Recieve output as comma seperated string.
            patterns =  self.main_chain.run(emails)

            #take the first pattern only
            patterns = patterns[0]
            
            return self._post_process_text(patterns)
        except Exception as excep:
            logging.error(f"Error while processing emails: {excep}")




       


