import os
import openai
import json

from create_dataset.type.entities import EntityData
from create_dataset.type.no_fake_timelines import Doc
from create_dataset.type.fake_news_dataset import DocForDataset

from create_dataset.gpt.classic_gpt import ClassicGPTResponseGetter

class FakenewsGPTResponseGetter(ClassicGPTResponseGetter):

    '''
    === For function calling: format fake news ===
    '''
    # Get 1st gpt response
    def get_gpt_response_fakenews_1st_step(self, messages: list, model_name='gpt-4', temp=1.0):
        openai.organization = os.environ['OPENAI_KUNLP']
        openai.api_key = os.environ['OPENAI_API_KEY_TIMELINE']

        response = openai.ChatCompletion.create(
            model=model_name,
            temperature=temp,
            messages=messages,
            request_timeout=180,
            functions=[
                self._format_fakenews_1st_info(),
            ],
            function_call='auto'
        )

        response_message = response['choices'][0]['message']
        assistant_message = {'role': 'assistant', 'content': response_message['content']}
        messages.append(assistant_message)

        if not response_message.get('function_call'):
            print('No function calling (fakenews_gpt.py).')
            return messages
        else:
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "format_fakenews_1st": self.format_fakenews_1st,
            }
            function_name = response_message['function_call']['name']
            print(f"Called function in fakenews_gpt.py: {function_name}")
            function_to_call = available_functions[function_name]
            try:
                function_args = json.loads(response_message['function_call']['arguments'])
            except json.decoder.JSONDecodeError as e:
                print(f"json.decoder.JSONDecodeError: {e}")
                print(response_message['function_call']['arguments'])

            if function_name == 'format_fakenews_1st':
                function_response = function_to_call(
                    position=function_args.get('position')
                )
                return function_response

    def format_fakenews_1st(self, position: int) -> int:
        return position

    def _format_fakenews_1st_info(self):
        function_info = {
            'name': 'format_fakenews_1st',
            'description': 'Convert the position of fake news into an integer type.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'position': {
                        'type': 'integer',
                        'description': 'The position of fake news. This must be a positive integer greater than 3.'
                    }
                },
                'required': ['position']
            }
        }
        return function_info


    # Get 2nd gpt response
    def get_gpt_response_fakenews_2nd_step(self, messages: list, model_name='gpt-4', temp=1.0):
        openai.organization = os.environ['OPENAI_KUNLP']
        openai.api_key = os.environ['OPENAI_API_KEY_TIMELINE']

        response = openai.ChatCompletion.create(
            model=model_name,
            temperature=temp,
            messages=messages,
            request_timeout=180,
            functions=[
                self._format_fakenews_2nd_info(),
            ],
            function_call='auto'
        )

        response_message = response['choices'][0]['message']
        assistant_message = {'role': 'assistant', 'content': response_message['content']}
        messages.append(assistant_message)

        if not response_message.get('function_call'):
            print('No function calling (fakenews_gpt.py).')
            return messages
        else:
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "format_fakenews_2nd": self.format_fakenews_2nd,
            }
            function_name = response_message['function_call']['name']
            print(f"Called function in fakenews_gpt.py: {function_name}")
            function_to_call = available_functions[function_name]
            try:
                function_args = json.loads(response_message['function_call']['arguments'])
            except json.decoder.JSONDecodeError as e:
                print(f"json.decoder.JSONDecodeError: {e}")
                print(response_message['function_call']['arguments'])

            if function_name == 'format_fakenews_2nd':
                function_response: DocForDataset = function_to_call(
                    headline=function_args.get('headline'),
                    short_description=function_args.get('short_description'),
                    date=function_args.get('date'),
                    content=function_args.get('content')
                )
                fake_news = function_response
                return fake_news

    def format_fakenews_2nd(self, headline, short_description, date, content) -> DocForDataset:
        fake_news: DocForDataset = {
            'ID': -1,
            'is_fake': True,
            'headline': headline,
            'short_description': short_description,
            'date': date,
            'content': content
        }
        return fake_news

    def _format_fakenews_2nd_info(self):
        function_info = {
            'name': 'format_fakenews_2nd',
            'description': 'Convert the information about generated fake news into a dictionary type.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'headline': {
                        'type': 'string',
                        'description': 'headline of fake news.'
                    },
                    'short_description': {
                        'type': 'string',
                        'description': 'short description of fake news.'
                    },
                    'date': {
                        'type': 'string',
                        'description': 'The date of fake news. This must be YYYY-MM-DD.'
                    },
                    'content': {
                        'type': 'string',
                        'description': 'content of fake news.'
                    }
                },
                'required': ['headline', 'short_description', 'date', 'content']
            }
        }
        return function_info
