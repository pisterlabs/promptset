import unittest
from unittest.mock import MagicMock, patch
import logging

from botengine_pytest import BotEnginePyTest
from locations.location import Location
import properties

import utilities.genai as genai
import openai

class TestGenAI(unittest.TestCase):

    @patch('openai.Completion.create')
    def test_genai_open_ai_response(self, completion_create_mock):
        
        # Initial setup
        botengine = BotEnginePyTest({})
        location_object = Location(botengine, 0)

        # Mock the OpenAI API
        completion = openai.Completion().create()
        completion.choices = [{"text": "Hi!"}]
        completion_create_mock.return_value = completion
        
        # Test the function
        response = genai.open_ai_response(botengine, location_object, {"prompt": genai.STOP_SEQUENCE_INPUT + "Hello" + genai.STOP_SEQUENCE_OUTPUT})
        assert response is not None
    
    @patch('botengine_pytest.BotEnginePyTest.get_secret', MagicMock(return_value='{"appname": "test", "appsecret": "__OPEN_AI_API_KEY__"}'))
    def test_genai_open_ai_response_debug(self):
        """
        Insert your OpenAI API key into the `appsecret` field to test the OpenAI API.
        This test will only run if the `appsecret` above has been replaced with a new string.
        """
        # Initial setup
        botengine = BotEnginePyTest({})
        location_object = Location(botengine, 0)
        secret = botengine.get_secret(genai.AWS_SECRET_NAME)
        import json
        secret = json.loads(secret)
        if secret["appsecret"] == "__OPEN_AI_API_KEY__":
            # Ignore this test
            return
        
        # Test the function
        response = genai.open_ai_response(botengine, location_object, {"prompt": genai.STOP_SEQUENCE_INPUT + "Hello" + genai.STOP_SEQUENCE_OUTPUT})
        assert response is not None