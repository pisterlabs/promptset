import os
import re

import openai

from gpt3.prompt import OPTIONAL_EXAMPLE, PROMPT_INIT, PROMTABLE_EXAMPLES, SLOTS_INIT

openai.api_key = os.getenv("OPENAI_API_KEY")

class UtteranceGenerator:
    EXAMPLES_TO_INCLUDE_IN_PROMPT = ['DecreaseBrightness', 'GetWeather', 'Greeting', 'BookRestaurant']
    STOP_SEQUENCE = '###'
    MAX_REQUESTS = 4

    def __init__(self, examples_to_include_in_prompt:list=None, gpt3_settings:dict={}):

        examples_to_include_in_prompt = examples_to_include_in_prompt or self.EXAMPLES_TO_INCLUDE_IN_PROMPT

        self.PROMPT = f'\n{self.STOP_SEQUENCE}\n'.join(
            [v.strip('\n') for k, v in PROMTABLE_EXAMPLES.items() if k in examples_to_include_in_prompt] + [PROMPT_INIT]
        ).strip('\n')

        self.gpt3_settings = {
            'engine': 'davinci',
            'temperature':0.85,
            'top_p':1, 
            'frequency_penalty':0.1,
            'presence_penalty':0,
            'best_of':1,
            'max_tokens':1000,
            'stop':[self.STOP_SEQUENCE],
        }
        self.gpt3_settings.update(gpt3_settings)

    def generate(self, intent:str, slots:list, n:int = 5, example:str=None, filter_na_slots:bool=True, max_length:int=200):
        """
        Args:
            intent: e.g. PlayMusic
            slots: list of possible slot, e.g. ['artist_name', 'song_name'],
            n: number of desired utterances (warning: it should be considered a guidance, the number of utterances may vary due to gpt3 outputs generating a different number and filtering) 
            example: an optional utterance example
            filter_na_slots: filter utterances that contain a slot not specified
            max_length: max character length of a *single* utterance
        """
        prompt_settings = {
            'intent': intent,
            'slots_init': SLOTS_INIT[bool(len(slots))].format(slots=', '.join(slots)),
            'n': n + bool(example),
        }
        prompt = self.PROMPT.format(**prompt_settings).strip('\n')

        if example:
            prompt += OPTIONAL_EXAMPLE.format(example=example)

        requests = 0
        utterances = []
        while len(utterances) < n and requests < self.MAX_REQUESTS:
            utterances += self._get_utterances(prompt, slots, filter_na_slots, max_length)
            requests += 1

        return utterances

    def _get_utterances(self, prompt, slots, filter_na_slots, max_length):
        response = openai.Completion().create(prompt=prompt, **self.gpt3_settings)

        utterances = [t.lstrip('- ') for t in response['choices'][0]['text'].strip('\n').split('\n')]

        if filter_na_slots:
            utterances_slots = [re.findall('\((.*)\)', utt) for utt in utterances]
            utterances = [utt for utt, utt_slots in zip(utterances, utterances_slots)
                            if all(utt_slot in slots for utt_slot in utt_slots)]
        
        if max_length:
            utterances = [utt for utt in utterances if len(utt) <= max_length]
        
        return utterances

generator = UtteranceGenerator()
