#!/usr/bin/python3
# -*- coding: utf-8 -*-

import requests
import pytest
import os
from openai import AzureOpenAI
import unittest
import testinfra
import httpx
from urllib.parse import urlsplit
from dotenv import load_dotenv

#
# Set Azure OpenAI Envriroment
#

# AZURE_OPENAI_ENDPOINT_URL = ""
# AZURE_OPENAI_KEY = ""
# DEPLOYMENT_NAME = ""

if load_dotenv(".env-aoai"):
    AZURE_OPENAI_ENDPOINT_URL = os.getenv('AZURE_OPENAI_ENDPOINT_URL')
    AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
    DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME')
else:
    print("Please set Azure OpenAI Environment Variables")


class TestAzureOpenAIEndpoints(unittest.TestCase):

    def setUp(self):
        self.host = testinfra.get_host("local://")
        self.aoai_netloc = urlsplit(AZURE_OPENAI_ENDPOINT_URL).netloc

    def test_to_aoai_endpoint_dns(self):
        aoai_endpoint = self.host.addr(self.aoai_netloc)
        self.assertTrue(aoai_endpoint.is_resolvable) # Equal to "getent ahosts 168.63.129.16"

    def test_to_aoai_endpoint_ip(self):
        aoai_endpoint = self.host.addr(self.aoai_netloc)
        self.assertTrue(aoai_endpoint.is_reachable) # Equal to "ping -W 1 -c 1 168.63.129.16"
        self.assertTrue(aoai_endpoint.port(80).is_reachable) # Equal to "nc -w 1 -z 168.63.129.16 80"
        self.assertTrue(aoai_endpoint.port(443).is_reachable) # Equal to "nc -w 1 -z 168.63.129.16 443"

    def test_to_aoai_endpoint_https(self):
        response = requests.get(AZURE_OPENAI_ENDPOINT_URL, verify=True)
        self.assertEqual(response.status_code, 404) # By de-fault, Azure OpenAI Endpoint is return 404 if there is no request

    def test_prompt(self):
        azure_client = AzureOpenAI(
                api_version = "2023-05-15",
                api_key = AZURE_OPENAI_KEY,
                azure_endpoint = AZURE_OPENAI_ENDPOINT_URL,
                http_client = httpx.Client(http2=True, verify=True)
            )

        response = azure_client.chat.completions.create(
            model = DEPLOYMENT_NAME,
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
                {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
                {"role": "user", "content": "Do other Azure AI services support this too?"}
            ],
        )
        # print(response.choices[0].message.content)
        assert response.usage.completion_tokens > 0


if __name__ == "__main__":
    unittest.main()
