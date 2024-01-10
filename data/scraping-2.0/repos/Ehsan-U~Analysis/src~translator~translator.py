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

from src.translator.prompts import translation_prompt
from langchain.output_parsers import CommaSeparatedListOutputParser

from src.logger import logging

import re

class Translator:
    """
    This class processes emails using a chain of language models to identify and categorize patterns in the emails. It connects to a language model and defines a processing chain for analyzing email content.
    """
    def __init__(self, user_settings:dict):
        self.connect_to_llm(user_settings["openAI_model_name"])
        self._initialize_translation_chain()

    def connect_to_llm(self, model_name:str):
        """
        establish connection to the llm
        """

        logging.info("Establishing connection to open ai models for translation")
        try:
            self._llm = ChatOpenAI(model_name=model_name, temperature=0)

            messages = [
                        SystemMessage(
                                content="You are a helpful assistant that is able to convert text from any language to English."
                        )
                        ]
            self._llm(messages)

        except Exception as excep:
            logging.error(f"Error establishing connection to llm for translation: {excep}")

    def _initialize_translation_chain(self):
        logging.info("Initializing translation chain")
        try:

            output_parser = CommaSeparatedListOutputParser()        
            self._chain = LLMChain(llm=self._llm, prompt=translation_prompt, output_key = "final_result", output_parser=output_parser, verbose=False)
        
        except Exception as excep:
            logging.error(f"Error initializing translation chain: {excep}")

    def _mark_titles(self, titles: list):
        marked_places = [i for i in range(len(titles)) if titles[i].strip() != ""]
        return marked_places

    def _preprocess_titles(self, titles: list):
        processed_titles = [titles[i].strip() for i in range(len(titles)) if titles[i].strip() != ""]
        return processed_titles
    
    def _postprocess_titles(self, titles, translated_titles, marked_indexes):
        for i, number in enumerate(marked_indexes):
            titles[number] = translated_titles[i]
        return titles

    def translate(self, company_name: str, titles: list):
        logging.info("Translating titles")
        try:
            #Store the indexes that has actual words in their position
            marked_indexes = self._mark_titles(titles)

            #Process all original list and remove empty spaces
            processed_titles = self._preprocess_titles(titles)
            if len(processed_titles) == 0:
                return titles

            #convert the list of titles into string and send to translation chain
            str_titles = "\n".join(processed_titles)
            translated_titles =  self._chain.run({"text":str_titles, "company": company_name})

            #Add the processed titles to the original list and return the translated information
            return self._postprocess_titles(titles, translated_titles, marked_indexes)

        except Exception as excep:
            logging.error(f"Error while translating emails: {excep}")




       


