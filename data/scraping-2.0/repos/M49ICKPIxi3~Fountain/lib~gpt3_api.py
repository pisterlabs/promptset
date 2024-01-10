
#test = 'nothing'
import json
import os
from dataclasses import dataclass

from enum import Enum
from glob import iglob


import openai


import threading

gpt3_api_lock = threading.Lock()

# TODO: Just delete this, hoping it's faster than messing w/ imports
class SingletonOptmized3(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with gpt3_api_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(SingletonOptmized3, cls).__call__(*args, **kwargs)
        return cls._instances[cls]



def _remove_empty(_dict):
    return {_key: _value for _key, _value in _dict.items() if _value is not None}


@dataclass
class CompletionPreset(object):
    name = None
    preset_type = 'default'
    engine = None
    model = None
    prompt = ''
    temperature = 0.25
    max_tokens = 333
    top_p = 1.0
    frequency_penalty = 0.45
    presence_penalty = 0.0
    echo = False
    n = 1
    logit_bias = None
    logprobs = None
    stop = None






    def _make_exclusive(self, _dict):  # Fine-Tune'd engines are called models, engines are the defaults
        if 'preset_type' in _dict:
            if _dict['preset_type'] == 'default':
                if 'model' in _dict:
                    del _dict['engine']
            else:
                if 'engine' in _dict:
                    del _dict['model']
        return _dict

    def to_preset_json(self):
        return self._make_exclusive({
            'name': self.name,
            'preset_type': self.preset_type,
            'engine': self.engine,  # Either this or model will be set
            'model': self.model,
            # 'prompt': self.prompt,  # I see no reason to include this in the preset json just yet
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'echo': False, #self.echo,
            'n': self.n,
            'logprobs': self.logprobs,  # 0-20
            'stop': self.stop,  # ['\n'] = stop after sent, ['\n\n'] = stop after paragraph
            'logit_bias': self.logit_bias
        })


    def to_args(self):
        return self._make_exclusive(self._remove_empty({
            'engine': self.engine,  # Either this or model will be set
            'model': self.model,
            'prompt': self.prompt,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'echo': False, #self.echo,
            'n': self.n,
            'logprobs': self.logprobs,  # 0-20
            'stop': self.stop,  # ['\n'] = stop after sent, ['\n\n'] = stop after paragraph
            'logit_bias': self.logit_bias
        }))

# TODO: WIP, just create these using the CompletionPreset object
def make_new_preset(name, engine_id):
    return CompletionPreset(**{
            'name': name,
            'preset_type': 'default',
            'engine': engine_id,
            'temperature': 0.25 if 'codex' in engine_id else 0.9,
            'max_tokens': 400 if 'codex' in engine_id else 250,
            'top_p': 1,
            'frequency_penalty': 0.55 if 'codex' in engine_id else 0.35,
            'presence_penalty': 0.0,
            'echo': False,
            'n': 1
        })

# TODO: WIP, just create these using the CompletionPreset object
def make_new_fine_tune_preset(name, model_id):
    return CompletionPreset(**{
            'name': name,
            'preset_type': 'fine_tune',
            'engine': model_id,
            'temperature': 0.9,
            'max_tokens': 250,
            'top_p': 1,
            'frequency_penalty': 0.35,
            'presence_penalty': 0.0,
            'echo': False,
            'n': 1
        })


# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

prompts_dir = '/Users/saya/PycharmProjects/writing_tools/main/code_generator/prompts/'
responses_dir = '/Users/saya/PycharmProjects/writing_tools/main/code_generator/responses/'

openai.api_key = os.getenv('OPENAI_API_KEY')

"""
TODO: Create a multi-level system, right now you can use a preset combined with a prompt
      but a more ideal UI is to use a preset and switch the prompt the preset uses. 
      That or keep a history of all prompts used with all presets.

"""


class GPT3API(metaclass=SingletonOptmized3):
    openai_api = openai
    presets = []

    def save_presets(self):
        return

    def completion(self, completion_parameters):
        _parameters = _remove_empty(completion_parameters)
        if 'name' in _parameters:
            del(_parameters['name'])
        if 'preset_type' in _parameters:
            del(_parameters['preset_type'])

        _parameters['timeout'] = 7

        response = self.openai_api.Completion.create(**_parameters)

        response_choice = response['choices'][0]
        text_output = response_choice['text']

        return text_output



def make_default_presets(presets_dir, engines_list):
    for engine in engines_list['data']:
        engine_id = engine['id']
        default_preset = {
            'engine': engine_id,
            'temperature': 0.25 if 'codex' in engine_id else 0.9,
            'max_tokens': 400 if 'codex' in engine_id else 250,
            'top_p': 1,
            'frequency_penalty': 0.55 if 'codex' in engine_id else 0.35,
            'presence_penalty': 0.0,
            'echo': False,
            'n': 1
        }

        file_path = presets_dir + '/' + engine_id + '.sublime-settings'
        with open(file_path, 'w+') as file:

            json.dump(
                default_preset,
                file,
                indent=4,
                ensure_ascii=True
            )
"""
def main(args=None):

    engines_list = openai.Engine.list()
    stop_here = ''

    presets_dir = '/Users/saya/PycharmProjects/fountain/presets/'

    presets = []
    for engine in engines_list['data']:
        engine_id = engine['id']
        default_preset = CompletionPreset(**{
            'name': engine_id,
            'preset_type': 'default',
            'engine': engine_id,
            'temperature': 0.25 if 'codex' in engine_id else 0.9,
            'max_tokens': 400 if 'codex' in engine_id else 250,
            'top_p': 1,
            'frequency_penalty': 0.55 if 'codex' in engine_id else 0.35,
            'presence_penalty': 0.0,
            'echo': False,
            'n': 1
        })
        file_path = presets_dir + engine_id + '.json'
        with open(file_path, 'w+') as file:
            json.dump(
                default_preset.to_preset_json(),
                file,
                indent=4,
                ensure_ascii=True
            )

    codex_api = GPT3API()

    logit_bias = {
            '2': -100,          #   "#"
            '1303': -100,       #   " #"
            '2235': -100,       #   "##"
            '22492': -100,      #   " ##"
            '21017': -100,      #   "###"
            '44386': -100,      #   " ###"
            '37811': -100,      #   "\"\"\""
            '37227': -100,       #   " \"\"\""
            '6738': -30         # from
    }

    _prompt_name = 'default'

    if args != None:
        _prompt_name = args[0]

    #prompt_text = codex_api.read_prompt(prompt_name=_prompt_name)

    completion_params = CompletionPreset(**{
        'engine': 'curie:ft-user-evm8sccfmua1zbbdrysa1pnf-2021-09-20-12-59-56',
        'temperature': 0.25,
        'max_tokens': 400,
        'top_p': 1,
        'frequency_penalty': 0.5,
        'presence_penalty': 0.0,
        'echo': True,
        'n': 1, #,
        'logit_bias': logit_bias
    })

    #resp_text = codex_api.completion(prompt_name=_prompt_name, codex_params=codex_params)

"""

"""
import sys

if __name__ == '__main__':
    main(sys.argv[1:])"""





