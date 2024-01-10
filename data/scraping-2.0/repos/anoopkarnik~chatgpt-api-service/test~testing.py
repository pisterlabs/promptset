'''
functions for testing all services and apis
'''

import unittest
import json
import os
import logging
import requests
import time
import json
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from main.services.chat import chat_with_gpt
from main.services.assistants import get_reply_by_assistant
from main.services.audio import create_text_to_speech,create_speech_to_text,translate_speech_to_text
from main.controllers.Controller import payload_controller
from openai import OpenAI
from app import create_app

logger = logging.getLogger(__name__)

# class TestChat(unittest.TestCase):
#     def __init__(self,*args,**kwargs):
#         self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
#     def test_chat_with_gpt(self):
#         client = self.client
#         message_body = {
#             "message":"Hello, how are you?",
#             "system_instructions":"You are a helpful assistant"
#         }
#         result = chat_with_gpt(message_body)
#         self.assertTrue(result['role'] == 'system')
#         self.assertTrue(result['text'] != '')
#         self.assertTrue(result['text'] != message_body['message'])

# class TestAssistants(unittest.TestCase):
#     def __init__(self,*args,**kwargs):
#         super(TestAssistants,self).__init__(*args,**kwargs)
#         self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
#         self.assitant_name='SOFTWARE_ENGINEER_MENTOR'
#     def test_get_reply_by_assistant(self):
#         message_body = {
#             "message":"Hello, how are you?",
#             "thread":"new",
#             "assistant_name":self.assitant_name
#         }
#         client = self.client
#         result = get_reply_by_assistant(message_body)
#         self.assertTrue(result['role'] == 'system')
#         self.assertTrue(result['text'] != '')
#         self.assertTrue(result['text'] != message_body['message'])

# class TestAudio(unittest.TestCase):
#     def test_create_text_to_speech(self):
#         message_body = {
#             "message":"Hello, how are you?",
#             "output_path":"data",
#             "file_name":"test.mp3"
#         }
#         create_text_to_speech(message_body)
#         self.assertTrue(os.path.exists(os.path.join(message_body['output_path'],message_body['file_name'])))

#     def test_create_speech_to_text(self):
#         message_body = {
#             "path":"data/test.mp3"
#         }
#         transcript = create_speech_to_text(message_body)
#         self.assertTrue(transcript['data']['text'] != '')

#     def test_translate_speech_to_text(self):
#         message_body = {
#             "path":"data/test.mp3"
#         }
#         transcript = translate_speech_to_text(message_body)
#         self.assertTrue(transcript['data']['text'] != '')

# class TestController(unittest.TestCase):
#     def test_health_check(self):
#         response = requests.get('http://0.0.0.0:8111/')
#         self.assertTrue(response.status_code == 200)
#         self.assertTrue(response.json()['status'] == 'success')

#     def test_chat_with_assistant(self):
#         message_body = {
#             "message":"Hello, how are you?",
#             "system_instructions":"You are a helpful assistant"
#         }
#         response = requests.post('http://0.0.0.0:8111/',json=message_body)
#         self.assertTrue(response.status_code == 200)
#         self.assertTrue(response.json()['role'] == 'system')
#         self.assertTrue(response.json()['text'] != '')
#         self.assertTrue(response.json()['text'] != message_body['message'])

#     def test_text_to_speech(self):
#         message_body = {
#             "message":"Hello, how are you?",
#             "output_path":"data",
#             "file_name":"test.mp3"
#         }
#         response = requests.post('http://0.0.0.0:8111/',json=message_body)
#         self.assertTrue(response.status_code == 200)
#         self.assertTrue(os.path.exists(os.path.join(message_body['output_path'],message_body['file_name'])))
#         os.remove(os.path.join(message_body['output_path'],message_body['file_name']))
    
#     def test_speech_to_text(self):
#         message_body = {
#             "path":"data/test.mp3"
#         }
#         response = requests.post('http://0.0.0.0:8111/',json=message_body)
#         self.assertTrue(response.status_code == 200)
#         self.assertTrue(response.json()['data']['text'] != '')

#     def test_translate(self):
#         message_body = {
#             "path":"data/test.mp3"
#         }
#         response = requests.post('http://0.0.0.0:8111/',json=message_body)
#         self.assertTrue(response.status_code == 200)
#         self.assertTrue(response.json()['data']['text'] != '')

if __name__ == '__main__':
    unittest.main()
    
                                 
                                 
                                 