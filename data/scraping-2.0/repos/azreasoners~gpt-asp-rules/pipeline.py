import json
import os
import pickle
import time

from clingo.control import Control
from clingo.symbol import parse_term
import pandas as pd
import openai

from api_keys import API_KEY, ORG_KEY

# clingo context used to define python functions in clingo
class Context:
    # get features/words from a string of space separated words
    def gen_feature(self, x):
        ret = []
        for term in str(x.string).split(' '):
            ret.append(parse_term(term))
        return ret

class Pipeline:
    def __init__(self, args):
        self.asp_program = ''
        ###########
        # GPT-3
        ###########
        self.org_key = ORG_KEY
        self.api_key = API_KEY
        self.engine = 'text-davinci-003'
        self.temperature = 0.
        self.max_tokens = 1500
        self.path_prompt = {} # store the mapping from kind (str) to path of prompt file (str)
        self.prompt = {} # a mapping from prompt kind (str) to the prompt (str)
        ###########
        # Cache
        ###########
        self.path_cache = {} # store the mapping from kind (str) to path of cache file (str)
        self.cache = {} # store the mapping from kind (str) to cached responses (dictionary)
        self.path_mistakes = 'mistakes.xlsx' # file to store the wrong pridictions
        self.mistakes = [] # store the wrong predictions

        for k,v in args.items():
            setattr(self, k, v)
        # initialze openai account
        if self.org_key:
            openai.organization = self.org_key
        openai.api_key = self.api_key

    def load_prompt(self):
        for kind in self.path_prompt:
            with open(self.path_prompt[kind], 'r', encoding='utf-8') as f:
                self.prompt[kind] = f.read().strip()

    def load_cache(self):
        for kind in self.path_cache:
            if os.path.isfile(self.path_cache[kind]):
                with open(self.path_cache[kind], 'r') as f:
                    self.cache[kind] = json.load(f)
            else:
                self.cache[kind] = {}

    def save_cache(self):
        for kind in self.path_cache:
            with open(self.path_cache[kind], 'w') as f:
                json.dump(self.cache[kind], f)

    # take a kind and replace (dictionary), return the GPT3 response
    def gen_response(self, kind, replace):
        # obtain the whole prompt
        prompt = self.prompt[kind]
        for k in replace:
            prompt = prompt.replace(k, replace[k])
        # generate and cache the response in cache if it's not cached before
        if prompt not in self.cache[kind]:
            try:
                if self.engine == 'gpt-4':
                    messages = [{'role': 'user', 'content': prompt}]
                    try:
                        self.cache[kind][prompt] = openai.ChatCompletion.create(
                            messages=messages,
                            model="gpt-4",
                            temperature=self.temperature,
                            max_tokens=self.max_tokens)
                    except:
                        print('GPT error.')
                else:
                    self.cache[kind][prompt] = openai.Completion.create(
                        prompt=prompt,
                        engine=self.engine,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens)
                self.save_cache()
            except Exception as e:
                print(e)
                breakpoint()
                self.cache[kind][prompt] = None
        if self.engine == 'gpt-4':
            return self.cache[kind][prompt]['choices'][0]['message']['content'].strip()
        return self.cache[kind][prompt]['choices'][0]['text'].strip()

    # take a kind and replace (dictionary), return the GPT3 response
    def gen_response_constraints(self, kind, replace):
        # obtain the whole prompt
        prompt = self.prompt[kind]
        for k in replace:
            prompt = prompt.replace(k, replace[k])
        # generate and cache the response in cache if it's not cached before
        if prompt not in self.cache[kind]:
            try:
                if self.engine == 'gpt-4':
                    # split prompt into different messages
                    general, ex1, ex2, ex3 = prompt.split('\n\nProblem ')
                    ex1, response1 = ex1.split('\n\nConstraints:\n')
                    ex2, response2 = ex2.split('\n\nConstraints:\n')
                    ex1 = 'Problem ' + ex1 + '\n\nConstraints:'
                    ex2 = 'Problem ' + ex2 + '\n\nConstraints:'
                    ex3 = 'Problem ' + ex3
                    messages = [
                        {'role': 'system', 'content': 'You are a semantic parser to turn clues in a problem into logical rules using only provided constants and predicates.'},
                        {'role': 'system', 'name': 'example_user', 'content': general},
                        {'role': 'system', 'name': 'example_assistant', 'content': 'Ok. I will only write constraints under the provided forms.'},
                        {'role': 'system', 'name': 'example_user', 'content': ex1},
                        {'role': 'system', 'name': 'example_assistant', 'content': response1},
                        {'role': 'system', 'name': 'example_user', 'content': ex2},
                        {'role': 'system', 'name': 'example_assistant', 'content': response2},
                        {'role': 'user', 'content': ex3},
                        ]
                    self.cache[kind][prompt] = openai.ChatCompletion.create(
                        messages=messages,
                        model="gpt-4",
                        temperature=self.temperature,
                        max_tokens=self.max_tokens)
                else:
                    self.cache[kind][prompt] = openai.Completion.create(
                        prompt=prompt,
                        engine=self.engine,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens)
                self.save_cache()
            except Exception as e:
                print(e)
                breakpoint()
                self.cache[kind][prompt] = None
        if self.engine == 'gpt-4':
            return self.cache[kind][prompt]['choices'][0]['message']['content'].strip()
        return self.cache[kind][prompt]['choices'][0]['text'].strip()

    # take a kind and replace (dictionary), return the GPT response
    def gen_response_bk(self, kind, replace):
        # obtain the whole prompt
        prompt = self.prompt[kind]
        for k in replace:
            prompt = prompt.replace(k, replace[k])
        # generate and cache the response in cache if it's not cached before
        if prompt not in self.cache[kind]:
            try:
                self.cache[kind][prompt] = openai.Completion.create(
                    prompt=prompt,
                    engine=self.engine,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens)
                self.save_cache()
            except Exception as e:
                print(e)
                breakpoint()
                self.cache[kind][prompt] = None
        return self.cache[kind][prompt]['choices'][0]['text'].strip()

    # use ASP (clingo) to find answer sets
    def gen_answer_set(self, program, opt=False):
        """
        Args:
            program (str): a string of ASP program
            opt (bool): if true, only optimal answer sets are returned
                        leave it to False when there is no weak constraint
        """
        clingo_control = Control(['0', '--warn=none', '--opt-mode=optN', '-t', '4'])
        models = []
        try:
            clingo_control.add('base', [], program)
            clingo_control.ground([('base', [])], context=Context())
        except:
            # breakpoint()
            return []
        if opt:
            clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)) if model.optimality_proven else None)
        else:
            clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)))
        models = [[str(atom) for atom in model] for model in models]
        return models

    def save_mistakes(self, mistake_cols):
        df = pd.DataFrame(self.mistakes, columns=mistake_cols)
        writer = pd.ExcelWriter(self.path_mistakes)
        df.to_excel(writer, sheet_name='results')
        for col_idx in range(2, 10):
            writer.sheets['results'].set_column(col_idx, col_idx, 40)
        writer.close()
