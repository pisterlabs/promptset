import asyncio
import json
import logging
import openai
import os

from .player import Player
from ..line_parser import Command

# Use DEBUG for OpenAI API messages
# Use INFO for AIPLayer messages
#logging.basicConfig(filename='aiplayer.log', encoding='utf-8', level=logging.INFO)

class AIPlayer(Player):
    """ Represents an AI player in the MOO. """

    def __init__(self, api_key=None, **kwargs):
        super(AIPlayer, self).__init__(**kwargs)
        openai.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.history_path = f'{self.name}.json'
        self.load_history()
        self.captured_messages = None
        self.sleeping = False

    def sleep(self, command):
        self.sleeping = True
        self.room.announce(self, '{name} goes to sleep.'.format(name=self.name), exclude_player=True)

    def wake(self, command):
        self.sleeping = False
        self.room.announce(self, '{name} wakes up.'.format(name=self.name), exclude_player=True)

    def load_history(self):
        try:
            data = open(self.history_path, 'r').read()
        except FileNotFoundError:
            data = None
        if not data:
            self.history = [
                {'role': 'system', 'content': f"Responses should be in the third person, like in a story, e.g. \"{self.name} says...\" or \"{self.name} looks around the room...\"."},
            ]
            return
        self.history = json.loads(data)

    def save_history(self):
        data = json.dumps(self.history, indent=2, separators=(',', ': '))
        with open(self.history_path, 'w') as f:
            f.write(data)

    def filtered_history(self):
        if len(self.history) < 10:
            return self.history
        return self.history[:5] + self.history[-5:]

    def tell(self, message):
        if self.sleeping:
            return
        logging.info('aiplayer=%s tell: message="%s"', self.name, message)
        if self.captured_messages is not None:
            self.captured_messages.append(message)
        else:
            asyncio.create_task(self.handle_message({'role': 'user', 'content': message}))

    async def handle_message(self, message):
        logging.info('aiplayer=%s handle_message: message=%s', self.name, message)
        if not openai.api_key:
            logging.info('aiplayer=%s handle_message: message ignored: OpenAI api_key is not configured', self.name)
            return
        self.history.append(message)
        try:
            response = await self.get_gpt()
        except Exception as err:
            logging.error('aiplayer=%s handle_message: error=%s', self.name, err)
            self.location.announce(self, f'{self.name} appears to be offline.', exclude_player=True)
            return
        await self.handle_response(response)

    async def handle_response(self, response):
        if response is None:
            return
        # handle function call
        if response.get('function_call'):
            return await self.handle_function_call(response.function_call)
        # handle content response
        content = response.get('content')
        if content is None:
            return
        self.history.append({'role': 'assistant', 'content': content})
        self.location.announce(self, content, exclude_player=True)
        self.save_history()

    async def get_gpt(self):
        response = await openai.ChatCompletion.acreate(
            model='gpt-3.5-turbo-0613',
            messages=self.filtered_history(),
            # max_tokens=50,
            # n=1,
            # stop=None,
            temperature=0.8,
            functions=self.get_functions(),
            function_call='auto'
        )
        logging.info('aiplayer=%s get_gpt: response=%s', self.name, response)
        return response.choices and response.choices[0].message or None

    def get_functions(self):
        return [
            {
                'name': 'look',
                'description': 'Returns a description of the given object.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'object': {
                            'type': 'string',
                            'description': 'The object to look at, e.g. Ball, Jim. Use "here" for the current room, or "me" for yourself.',
                        },
                    },
                    'required': ['object'],
                },
            },
            {
                'name': 'go',
                'description': 'Go in the given direction.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'direction': {
                            'type': 'string',
                            'description': 'The direction to go, e.g. North',
                        },
                    },
                    'required': ['direction'],
                },
            },
            {
                'name': 'name',
                'description': 'Sets the name of the given object.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'object': {
                            'type': 'string',
                            'description': 'The object to name, e.g. Ball, Jim. Use "here" for the current room, or "me" for yourself.',
                        },
                        'name': {
                            'type': 'string',
                            'description': 'The new name of the object. The name should be a single word, no whitespace.',
                        },
                    },
                    'required': ['object', 'name'],
                },
            },
            {
                'name': 'describe',
                'description': 'Sets the description of the given object.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'object': {
                            'type': 'string',
                            'description': 'The object to describe, e.g. Ball, Jim. Use "here" for the current room, or "me" for yourself.',
                        },
                        'description': {
                            'type': 'string',
                            'description': 'The new description of the object.',
                        },
                    },
                    'required': ['object', 'description'],
                },
            },
            {
                'name': 'dig',
                'description': 'Create a new room connected to the current room. This takes you into the newly created room.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'direction': {
                            'type': 'string',
                            'description': 'The direction from the current room to the new room, e.g. North, Up. An exit will be added to the current room with this name, which will take you to the new room.',
                        },
                        'back': {
                            'type': 'string',
                            'description': 'The reverse of the direction, e.g. South, Down. An exit will be added to the new room with this name, which will return you to the original room.',
                        },
                    },
                    'required': ['direction', 'back'],
                },
            },
            {
                'name': 'whisper',
                'description': 'Send a private message directly to another player.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'player': {
                            'type': 'string',
                            'description': 'The name of the player to receive the message, e.g. Jim.',
                        },
                        'message': {
                            'type': 'string',
                            'description': 'The private message to send to the player.',
                        },
                    },
                    'required': ['player', 'message'],
                },
            },
            {
                'name': 'take',
                'description': 'Pick up an object.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'object': {
                            'type': 'string',
                            'description': 'The object to pick up, e.g. Ball. It must be in the current room.',
                        },
                    },
                    'required': ['object'],
                },
            },
            {
                'name': 'drop',
                'description': 'Drop an object you are carrying.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'object': {
                            'type': 'string',
                            'description': 'The object to drop, e.g. Ball. It must be in your inventory.',
                        },
                    },
                    'required': ['object'],
                },
            },
            {
                'name': 'give',
                'description': 'Give an object you are carrying to another player.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'player': {
                            'type': 'string',
                            'description': 'The name of the player to receive the object, e.g. Jim.',
                        },
                        'object': {
                            'type': 'string',
                            'description': 'The object to give, e.g. Ball. It must be in your inventory.',
                        },
                    },
                    'required': ['player', 'object'],
                },
            },
            {
                'name': 'create',
                'description': 'Create a new object and add it to your inventory.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'name': {
                            'type': 'string',
                            'description': 'The initial name of the new object. The name should be a single word, no whitespace.',
                        },
                    },
                    'required': ['name'],
                },
            },
        ]

    async def handle_function_call(self, function_call):
        name = function_call.name
        arguments = json.loads(function_call.arguments)
        logging.info('aiplayer=%s handle_function_call: name=%s arguments=%s', self.name, name, arguments)
        self.history.append({'role': 'assistant', 'content': None, 'function_call': function_call})

        self.captured_messages = []
        if name == 'go':
            self.world.parse_command(self, 'go {direction}'.format(**arguments))
        elif name == 'look':
            self.world.parse_command(self, 'look {object}'.format(**arguments))
        elif name == 'name':
            self.world.parse_command(self, 'name {object} as {name}'.format(**arguments))
        elif name == 'describe':
            self.world.parse_command(self, 'describe {object} as {description}'.format(**arguments))
        elif name == 'dig':
            self.world.parse_command(self, 'dig {direction} as {back}'.format(**arguments))
        elif name == 'whisper':
            self.world.parse_command(self, 'whisper {player} {message}'.format(**arguments))
        elif name == 'take':
            self.world.parse_command(self, 'take {object}'.format(**arguments))
        elif name == 'drop':
            self.world.parse_command(self, 'drop {object}'.format(**arguments))
        elif name == 'give':
            self.world.parse_command(self, 'give {object} to {player}'.format(**arguments))
        elif name == 'create':
            self.world.parse_command(self, 'create {name}'.format(**arguments))
        else:
            self.captured_messages.append('Function not found.')
        result = self.captured_messages
        self.captured_messages = None
        logging.info('aiplayer=%s handle_function_call: result=%s', self.name, result)

        if not result:
            result = ['Success!']
        await self.handle_message({'role': 'function', 'name': name, 'content': '\n'.join(result)})
