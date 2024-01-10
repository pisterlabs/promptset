import asyncio
import openai
import os
import json
from loguru import logger

# class that wraps the OpenAI client and Azure clients
class AiClient:
    def __init__(self, api_key = None, azure_endpoint = None, api_version = None, azure_ad_token = None, azure_ad_token_provider = None):
        # check environment variables and use those if explicit values are not passed in
        # API key
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        # Azure specific        
        if azure_endpoint is None:
            azure_endpoint = os.getenv("OPENAI_AZURE_ENDPOINT")

        # api_version
        if api_version is None:
            api_version = os.getenv("OPENAI_API_VERSION")

        # azure_ad_token
        if azure_ad_token is None:
            azure_ad_token = os.getenv("OPENAI_AZURE_AD_TOKEN")

        # azure_ad_token_provider
        if azure_ad_token_provider is None:
            azure_ad_token_provider = os.getenv("OPENAI_AZURE_AD_TOKEN_PROVIDER")

        # keep track of the values we're using
        self._api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.azure_ad_token = azure_ad_token
        self.azure_ad_token_provider = azure_ad_token_provider

        # if the azure_endpoint is set, we're an azure client
        if self.azure_endpoint is not None:
            self.is_azure = True
            self.aclient = openai.AsyncAzureOpenAI(api_key=api_key, azure_endpoint=self.azure_endpoint, api_version=self.api_version, azure_ad_token=self.azure_ad_token, azure_ad_token_provider=self.azure_ad_token_provider)
            self.client = openai.AzureOpenAI(api_key=api_key, azure_endpoint=self.azure_endpoint, api_version=self.api_version, azure_ad_token=self.azure_ad_token, azure_ad_token_provider=self.azure_ad_token_provider)
        else:
            self.is_azure = False
            self.aclient = openai.AsyncOpenAI(api_key=api_key)
            self.client = openai.OpenAI(api_key=api_key)
    
    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        self._api_key = value
        self.aclient.api_key = value
        self.client.api_key = value
