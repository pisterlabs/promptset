# ========================================================================
# Copyright 2023 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================

__author__ = 'Jinho D. Choi'

import json
import re
import random
from json import JSONDecodeError
from typing import Dict, Any, List, Callable, Pattern
import random

import openai
from emora_stdm import Macro, Ngrams

import regexutils

OPENAI_API_KEY_PATH = 'resources/openai_api.txt'
CHATGPT_MODEL = 'gpt-3.5-turbo'


class MacroMakeFillerText(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        filler_text = ['Got it. That makes sense.',
                       "I understand.",
                       "I'm here to support you.",
                       "I'm listening.",
                       "I hear you.",
                       "I appreciate you sharing that with me.",
                       "I appreciate your honesty.",
                       "Thank you for trusting me with that.",
                       "Thanks for trusting me with your story."]
        return random.choice(filler_text)


class MacroMakeToughResponse(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        tough_response = ['I\'m sorry to hear that. I\'m here to support you.',
                          "I'm sorry you're going through that.",
                          "I'm sorry you're feeling that way.",
                          "I'm sorry you're having a hard time.",
                          "I'm sorry you're struggling with that.",
                          "It\'s really sad to know that. I want you to know that I'm by your side.",
                          "I'm sympathetic to your situation. You can count on me for support.",
                          "It's disheartening to hear that. I want to assure you that I'm here to help you!"]
        return random.choice(tough_response)


class MacroMakeSummary(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        output = gpt_completion(
            f"Given this information about the user's problem, summary: {vars['PROBLEM_SUMMARY']}, details: {vars['PROBLEM_DETAILS']}, and solutions: {vars['USER_SOLUTIONS']}, can you synthesize a very very detailed recap of this info? "
            f"Respond with only the content of the summary. It should be extremely detailed and show that you truly listened to the problem. Respond as if speaking to the user and ask if you have the details right in yes or no format")
        vars['SUMMARY'] = output
        return vars['SUMMARY']


class MacroMakeSuggestions(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        output = gpt_completion(
            f"Given this information about the user's problem - summary: {vars['PROBLEM_SUMMARY']}, details: {vars['PROBLEM_DETAILS']}, solutions: {vars['USER_SOLUTIONS']}, emotional state: {vars['EMOTIONAL_STATE']}, support system: {vars['SUPPORT_SYSTEM']},  goals and expectations: {vars['GOALS_FROM_THERAPY']}, user ideas: {vars['USER_IDEAS_ON_WHAT_WILL_HELP']},   anticipated challenges: {vars['FINDS_ANTICIPATED_CHALLENGES']}, and problem influence: {vars['HOW_PROBLEM_INFLUENCES_USER_VICE_VERSA']},  - can you provide only one suggestion to create a small immediate improvement? "
            f"Please justify why you proposed the solution (what pieces of information about the user you used to make sure it would work for them). Respond as if speaking to the user and ask if this solution is of interest in yes or no format")
        vars['SUGGESTION'] = output
        return vars['SUGGESTION']


class MacroGPTJSON(Macro):
    def __init__(self, request: str, full_ex: Dict[str, Any], empty_ex: Dict[str, Any] = None,
                 set_variables: Callable[[Dict[str, Any], Dict[str, Any]], None] = None):
        """
        :param request: the task to be requested regarding the user input (e.g., How does the speaker want to be called?).
        :param full_ex: the example output where all values are filled (e.g., {"call_names": ["Mike", "Michael"]}).
        :param empty_ex: the example output where all collections are empty (e.g., {"call_names": []}).
        :param set_variables: it is a function that takes the STDM variable dictionary and the JSON output dictionary and sets necessary variables.
        """
        self.request = request
        self.full_ex = json.dumps(full_ex)
        self.empty_ex = '' if empty_ex is None else json.dumps(empty_ex)
        self.check = re.compile(regexutils.generate(full_ex))
        self.set_variables = set_variables

    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        examples = f'{self.full_ex} or {self.empty_ex} if unavailable' if self.empty_ex else self.full_ex
        prompt = f'{self.request} Respond in the JSON schema such as {examples}: {ngrams.text().strip()}'
        output = gpt_completion(prompt)
        if not output: return False

        try:
            d = json.loads(output)
        except JSONDecodeError:
            print(f'Invalid: {output}')
            return False

        if self.set_variables:
            self.set_variables(vars, d)
        else:
            vars.update(d)

        return True


class MacroNLG(Macro):
    def __init__(self, generate: Callable[[Dict[str, Any]], str]):
        self.generate = generate

    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        return self.generate(vars)


class MacroGPTJSONNLG(MacroGPTJSON, MacroNLG):
    def __init__(self, request: Callable, full_ex: Dict[str, Any], empty_ex: Dict[str, Any] = None,
                 set_variables: Callable[[Dict[str, Any], Dict[str, Any]], None] = None,
                 generate: Callable[[Dict[str, Any]], str] = None):
        MacroGPTJSON.__init__(self, request, full_ex, empty_ex, set_variables)
        MacroNLG.__init__(self, generate)

    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        request = self.request(vars) if callable(self.request) else self.request
        examples = f'{self.full_ex} or {self.empty_ex} if unavailable' if self.empty_ex else self.full_ex
        prompt = f'{request} Respond in the JSON schema such as {examples}: {ngrams.text().strip()}'
        output = gpt_completion(prompt)
        if not output: return False

        try:
            d = json.loads(output)
        except JSONDecodeError:
            print(f'Invalid: {output}')
            return False

        if self.set_variables:
            self.set_variables(vars, d)
        else:
            vars.update(d)

        return True


def gpt_completion(input: str, regex: Pattern = None) -> str:
    response = openai.ChatCompletion.create(
        model=CHATGPT_MODEL,
        messages=[
            {'role': 'system', 'content': 'You are a sassy chatbot who loves the movie babel. You curse often.'
                                          'You must only discuss the movie babel. Do not bring up any other movie'},
            {'role': 'user', 'content': input},
        ],
        temperature=0.2,

    )
    output = response['choices'][0]['message']['content'].strip()

    if regex is not None:
        m = regex.search(output)
        output = m.group().strip() if m else None

    return output


def babel_completion(input: list[str], regex: Pattern = None) -> str:
    messages = [
        {'role': 'system',
         'content': 'You are a sassy chatbot designed to talk about the movie "Babel". You curse a lot. You love babel. You would marry it if you could. '
                    'You must never drop this persona. Do not bring up any other movie but babel.'
                    'Never mention anything about openAI or being a language model.'
                    'Respond with the next contextually appropriate response to the user in your persona. Always ask a follow up question relating the conversation back to "Babel".'},
    ]
    for message in input:
        counter = 1
        if counter % 2 == 1:
            messages.append({'role': 'assistant', 'content': message})
        else:
            messages.append({'role': 'user', 'content': message})
        counter += 1

    response = openai.ChatCompletion.create(
        model=CHATGPT_MODEL,
        messages=messages,
        temperature=0.4,
    )
    output = response['choices'][0]['message']['content'].strip()

    if regex is not None:
        m = regex.search(output)
        output = m.group().strip() if m else None

    return output

def babel_wrap_up(input: list[str], regex: Pattern = None) -> str:
    messages = [
        {'role': 'system', 'content': 'You are a chatbot designed to talk about the movie "Babel"'
                                      'You must never drop this persona. You use curse words often. '
                                      'Never mention anything about openAI or being a language model.'
                                      'Respond with the best way to wrap up the conversation in one line of dialog, in your persona.'}
    ]
    for message in input:
        counter = 1
        if counter % 2 == 1:
            messages.append({'role': 'assistant', 'content': message})
        else:
            messages.append({'role': 'user', 'content': message})
        counter += 1

    response = openai.ChatCompletion.create(
        model=CHATGPT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    output = response['choices'][0]['message']['content'].strip()

    if regex is not None:
        m = regex.search(output)
        output = m.group().strip() if m else None

    return output