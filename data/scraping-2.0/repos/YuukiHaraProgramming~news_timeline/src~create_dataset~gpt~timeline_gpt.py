import os
import openai
import json

from typing import Tuple

from create_dataset.type.entities import EntityData
from create_dataset.type.no_fake_timelines import Doc

from create_dataset.gpt.classic_gpt import ClassicGPTResponseGetter

class TimelineGPTResponseGetter(ClassicGPTResponseGetter):
    '''
    === For function calling: format timeline ===
    '''
    # Get gpt response
    def get_gpt_response_timeline(self, messages: list, model_name='gpt-4', temp=1.0):
        openai.organization = os.environ['OPENAI_KUNLP']
        openai.api_key = os.environ['OPENAI_API_KEY_TIMELINE']

        response = openai.ChatCompletion.create(
            model=model_name,
            temperature=temp,
            messages=messages,
            request_timeout=180,
            functions=[
                self._format_timeline_info(),
            ],
            function_call='auto'
        )

        response_message = response['choices'][0]['message']
        assistant_message = {'role': 'assistant', 'content': response_message['content']}
        messages.append(assistant_message)

        if not response_message.get('function_call'):
            print('No function calling (timeline_gpt.py).')
            return messages
        else:
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "fortmat_timeline": self.fortmat_timeline,
            }
            function_name = response_message['function_call']['name']
            print(f"Called function in get_gpt_response.py: {function_name}")
            function_to_call = available_functions[function_name]
            try:
                function_args = json.loads(response_message['function_call']['arguments'])
            except json.decoder.JSONDecodeError as e:
                print(f"json.decoder.JSONDecodeError: {e}")
                print(response_message['function_call']['arguments'])

            if function_name == "fortmat_timeline":
                function_response, IDs_from_gpt = function_to_call(
                    number=function_args.get('number'),
                    IDs=function_args.get('IDs'),
                    reasons=function_args.get('reasons'),
                )
                return function_response, IDs_from_gpt

    def set_docs_num_in_1timeline(self, value: int):
        # self.__dict__['__docs_num_in_1timeline'] = value
        self.__docs_num_in_1timeline = value

    def set_entity_info(self, value: EntityData):
        self.__entity_info = value

    def fortmat_timeline(self, number: int, IDs: list[int], reasons: list[str]) -> Tuple[list[Doc], list[int]]:
        self.set_docs_num_in_1timeline(number)
        if not (len(IDs) == len(reasons) == self.__docs_num_in_1timeline):
            print(f"number: {number}")
            print(f"IDs: {IDs}")
            print(f"reasons: {reasons}")
            print('Input ERROR! (format_timeline in get_gpt_response.py)')
            return [], IDs
        else:
            timeline: list[Doc] = []
            for i in range(self.__docs_num_in_1timeline):
                doc_ID = IDs[i]
                get_doc_info_by_id = lambda: next((item for item in self.__entity_info['docs_info']['docs'] if item['ID'] == doc_ID), None)
                doc_info_dict = get_doc_info_by_id()
                headline, short_description, date, content = doc_info_dict['headline'], doc_info_dict['short_description'], doc_info_dict['date'], doc_info_dict['content']
                doc: Doc = {
                    'ID': IDs[i],
                    'is_fake': False,
                    'document': f"{headline}: {short_description}",
                    'headline': headline,
                    'short_description': short_description,
                    'date': date,
                    'content': content,
                    'reason': reasons[i]
                }

                timeline.append(doc)
            return timeline, IDs

    def _format_timeline_info(self):
        function_info = {
            'name': 'fortmat_timeline',
            'description': 'Convert the generated timeline information into a dictionary type.',
            'parameters': {
                # https://json-schema.org/understanding-json-schema/reference/array
                'type': 'object',
                'properties': {
                    'number': {
                        'type': 'integer',
                        'description': 'The number of documents.'
                    },
                    'IDs': {
                        'type': 'array',
                        'items': {'type': 'integer'},
                        'description': 'A list of Document ID.'
                    },
                    'reasons': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'A list of REASONS and STATEMENTS.'
                    }
                },
                'required': ['number', 'IDs', 'reasons']
            },
        }
        return function_info
