import logging
import os
from typing import Any, Dict

import openai

from src.constants.system_message_constants import (
    CATEGORIZATION_SYSTEM_MESSAGE,
    CHECK_RESPONSE_SYSTEM_MESSAGE,
    RESPONSE_SYSTEM_MESSAGE,
    TRANSLATION_SYSTEM_MESSAGE,
)
from src.utils.data_processor import DataProcessor

from .openai_api import OpenAI_API


class CustomerRequest:
    def __init__(self, request_text: str, openai_api: OpenAI_API) -> None:
        """
        Initialize a CustomerRequest.

        :param request_text: The customer request text.
        :param openai_api: An instance of OpenAI_API class.
        """
        self.request_text = request_text
        self.translated_text = None
        self.category = None
        self.response = None
        self.is_harmful = None
        self.openai_api = openai_api

        # Get a logger instance
        self.logger = logging.getLogger(__name__)

    def translate(self) -> None:
        """
        Translate the request text to English.
        """
        system_message = TRANSLATION_SYSTEM_MESSAGE
        user_message = self.request_text
        llm_result = self.openai_api.call_llm(system_message, user_message)
        self.translated_text = llm_result.choices[0].message.content

    def categorize(self) -> None:
        """
        Categorize the translated request text.
        """
        system_message = CATEGORIZATION_SYSTEM_MESSAGE
        user_message = self.translated_text
        llm_result = self.openai_api.call_llm(system_message, user_message)
        self.category = llm_result.choices[0].message.content

    def formulate_response(self) -> None:
        """
        Formulate a response to the translated request text.
        """
        # Create a DataProcessor instance
        data_processor = DataProcessor(
            input_dir="data/category_contexts", output_dir="output"
        )

        # Load category context
        category_context_file = f"{self.category}.txt"
        full_file_path = os.path.join(
            data_processor.input_dir, category_context_file
        )
        category_context = data_processor.load_text_file(full_file_path)

        # Include category context in the system message
        system_message = (
            RESPONSE_SYSTEM_MESSAGE
            + f"""
        {self.category}
        Here is some additional context:
        {category_context}
        """
        )
        user_message = self.translated_text
        llm_result = self.openai_api.call_llm(system_message, user_message)
        self.response = llm_result.choices[0].message.content

    def check_response(self) -> None:
        """
        Perform moderation and correctness checks on the formulated response.
        """
        # Moderation check
        moderation_response = openai.Moderation.create(input=self.response)
        moderation_result = moderation_response["results"][0]

        # Correctness check
        system_message = CHECK_RESPONSE_SYSTEM_MESSAGE
        user_message = f"""
        Customer Request: {self.translated_text} 
        ###
        Category Context: {self.category}
        ###
        Assistant Response: {self.response}"""
        check_response = self.openai_api.call_llm(
            system_message.format(user_message), user_message
        )
        correctness_result = check_response.choices[0].message.content

        self.is_harmful = {
            "moderation_result": moderation_result,
            "correctness_result": correctness_result,
        }

    def process_request(self) -> None:
        """
        Process the request: translate it, categorize it, formulate a response, and perform moderation and
        correctness checks on the response.
        """
        self.translate()
        self.logger.info(f"Translation: {self.translated_text}")

        self.categorize()
        self.logger.info(f"Category: {self.category}")

        self.formulate_response()
        self.logger.info(f"Response: {self.response}")

        self.check_response()
        self.logger.info(
            f"Moderation and correctness check: {self.is_harmful}"
        )
