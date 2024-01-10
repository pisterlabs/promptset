# auth.py

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI


@dataclass
class AuthHandle:
    # TO DO: handle selecting model. Default to GPT4, but select 3.5 if not available.

    logger = logging.getLogger(__name__)

    def configure(self):
        self.logger.debug("Loading env variables")
        load_dotenv()
        self.is_configured()

    def is_configured(self):
        key = os.getenv("OPENAI_API_KEY")
        if key != None:
            self.api_key = key
            self.logger.debug("Authenticated")
        else:
            self.logger.error("Invalid or missing api key")

    def get_available_models(self):
        client = OpenAI(api_key=self.api_key)

        models = client.models.list()
        return [model.id for model in models]
