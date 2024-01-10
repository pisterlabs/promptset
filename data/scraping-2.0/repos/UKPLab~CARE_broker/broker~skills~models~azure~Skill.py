""" Skill for OpenAI Azure Client

This skill is a simple wrapper for the OpenAI Azure Client.

Documentation Azure Client
https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Cpython-new&pivots=programming-language-python

Author: Dennis Zyska
"""

import os

from SkillSimple import SkillSimple
from openai import AzureOpenAI


class Skill(SkillSimple):
    """
    Skill for OpenAI API
    """

    def __init__(self, name):
        super().__init__(name)
        self.description = "This is a skill for the OpenAI Azure API"
        self.client = None
        self.model = os.environ.get('OPENAI_MODEL')

    def init(self):
        """
        Initialize Open AI Connection
        :return:
        """
        self.client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_KEY"),
            api_version=os.environ.get("API_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
        )

    def execute(self, data):
        """
        Execute a request to the OpenAI API
        :param data:
        :return:
        """
        response = self.client.chat.completions.create(
            model=self.model,  # model = "deployment_name".
            messages=data['messages'],
        )

        output = {
            "choices": [{
                "finish_reason": c.finish_reason,
                "index": c.index,
                "logprops": c.logprobs,
                "message": c.message.__dict__
            } for c in response.choices],
        }

        stats = {
            "id": response.id,
            "model": response.model,
            "object": response.object,
            "fingerprint": response.system_fingerprint,
            "usage": response.usage.__dict__,
        }
        # return None, None
        return output, stats

    def get_input(self):
        """
        Get the input schema
        :return:
        """
        return {
            'data': {
                'messages': {
                    'type': 'array',
                    'items': {
                        'type': "object",
                        "properties": {
                            "content": {
                                "type": 'string',
                            },
                            'role': {
                                "type": 'string',
                            },
                        }
                    },
                },
            },
            'example': {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, who are you?"},
                ]
            }
        }

    def get_output(self):
        """
        Get the output schema
        :return:
        """
        return {
            'data': {
                'choices': {
                    'type': 'array',
                    'items': {
                        'type': "object",
                        "properties": {
                            "finish_reason": {
                                "type": 'string',
                            },
                            'index': {
                                "type": 'integer',
                            },
                            'logprops': {
                                "type": 'object',
                            },
                            'message': {
                                "type": 'object',
                                "properties": {
                                    "message": {
                                        "type": 'string',
                                    },
                                    'role': {
                                        "type": 'string',
                                    },
                                    'function_call': {
                                        "type": 'object',
                                    },
                                    'tool_calls': {
                                        "type": 'object',
                                    },
                                }
                            },
                        }
                    },
                }
            },
            'stats': {
                'type': 'object',
                'properties': {
                    'id': {
                        "type": "string"
                    },
                    'model': {
                        "type": "string"
                    },
                    'object': {
                        "type": "string"
                    },
                    'usage': {
                        "type": "object",
                        "properties": {
                            "completion_tokens": {
                                "type": "integer"
                            },
                            "prompt_tokens": {
                                "type": "integer"
                            },
                            "total_tokens": {
                                "type": "integer"
                            }
                        }
                    }
                }
            },
            'example': {
                'choices': [
                    {'finish_reason': 'stop', 'index': 0, 'logprops': None, 'message': {
                        'content': "Hello! I'm an AI digital assistant designed to help answer questions, "
                                   "provide recommendations, and assist with various tasks. "
                                   "How can I assist you today?",
                        'role': 'assistant', 'function_call': None, 'tool_calls': None}}
                ],
                'stats': {
                    'result': {'id': 'chatcmpl-<id>', 'model': self.model,
                               'object': 'chat.completion',
                               'fingerprint': None,
                               'usage': {'completion_tokens': 30, 'prompt_tokens': 23, 'total_tokens': 53}
                               }
                }
            }
        }
