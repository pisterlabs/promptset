from typing import Dict, List, Tuple
from parser import constructionsplit
from openai import OpenAI
from constraint import check_constraint
import re

client = OpenAI(api_key="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")

class task:
    def __init__(self, query:List[str], passed_variable = {}, n = 1):
        self.scope = {}
        self.scope.update(passed_variable)
        self.trace = ''
        parsed_query = constructionsplit(query)
        self.query_body = parsed_query[0]
        self.output = parsed_query[1]
        self.constraints = parsed_query[2]
        self.n = n

    def run(self) -> Dict:
        pattern_brackets = re.compile(r'\[[a-zA-Z0-9]+?\]')
        pattern_curlybraces = re.compile(r'\{[a-zA-Z0-9]+?\}')
        for s in self.query_body:
            if pattern_brackets.search(s) != None:
                variablename = pattern_brackets.search(s).group(0).strip('\[\] ')
                constraint = ''
                for c in self.constraints:
                    if c.find(variablename) != -1:
                        constraint = c
                if constraint != '':
                    i = 0
                    step = {}
                    ith = {}
                    generated_sequence = ''
                    if constraint.find('stop_at') == -1:
                        generated_sequence = self.next_step(constraint, generated_sequence, i, step, ith, 1, True)
                    else:
                        generated_sequence = self.next_step_stop_at(constraint, generated_sequence, i, step, ith, 1, True)
                else:
                    model_output = client.completions.create(
                        model = "gpt-3.5-turbo-instruct",
                        prompt = self.trace,
                        logprobs = self.n
                    )
                    generated_sequence = model_output.choices[0].text
                self.trace += generated_sequence
                self.scope[variablename] = generated_sequence
            elif pattern_curlybraces.search(s) != None:
                variablename = pattern_curlybraces.search(s).group(0).strip('\{\} ')
                self.trace += self.scope[variablename]
            else:
                self.trace += s
        return_variable = {}
        self.output = self.output.strip('\[\] ')
        if self.output != '':
            variablename = self.output
            constraint = ''
            for c in self.constraints:
                if c.find(variablename) != -1:
                    constraint = c
            if constraint != '':
                i = 0
                step = {}
                ith = {}
                generated_sequence = ''
                if constraint.find('stop_at') == -1:
                    generated_sequence = self.next_step(constraint, generated_sequence, i, step, ith, 1, True)
                else:
                    generated_sequence = self.next_step_stop_at(constraint, generated_sequence, i, step, ith, 1, True)
            else:
                model_output = client.completions.create(
                    model = "gpt-3.5-turbo-instruct",
                    prompt = self.trace,
                    logprobs = self.n
                )
                generated_sequence = model_output.choices[0].text
            return_variable[self.output] = generated_sequence
        print(self.trace)
        return return_variable

    def next_step(self, constraint:str, generated_sequence:str, i:int, step:Dict, ith:Dict, ith_i = 1, new = True) -> str:
        if new == True:
            next_token = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=self.trace + generated_sequence,
                logprobs = self.n,
                max_tokens = 1
            )
            i += 1
            if next_token.choices[0].logprobs.top_logprobs == []:
                return generated_sequence
            step[i] = list(next_token.choices[0].logprobs.top_logprobs[0].keys())
            ith[i] = ith_i
        else:
            ith[i] = ith_i
        if ith_i == self.n + 1 and check_constraint(constraint, generated_sequence + step[i][ith_i - 2]) == True:
            generated_sequence += step[i][ith_i - 2]
            return generated_sequence
        if ith_i == self.n + 1 and i == 1:
            raise Exception('The language model cannot produce a sequence that satisfies the constraint: ' + constraint)
        elif ith_i == self.n + 1 and i > 1:
            i -= 1
            generated_sequence = self.sequence(i, step, ith)
            return self.next_step(constraint, generated_sequence, i, step, ith, ith[i] + 1, False)
        passed = check_constraint(constraint, generated_sequence + step[i][ith_i - 1])
        if passed == True and step[i][ith_i - 1] != '|<endoftext>|':
            generated_sequence += step[i][ith_i - 1]
            return self.next_step(constraint, generated_sequence, i, step, ith, 1, True)
        elif passed == True and step[i][ith_i - 1] == '|<endoftext>|':
            return generated_sequence
        elif passed == False and check_constraint(constraint, generated_sequence + step[i][ith_i - 2]) == True:
            generated_sequence += step[i][ith_i - 2]
            return generated_sequence
        elif passed == False and ith_i <= self.n:
            return self.next_step(constraint, generated_sequence, i, step, ith, ith[i] + 1, False)
        else:
            raise Exception('Execution error!')

    def sequence(self, i:int, step:Dict, ith:Dict) -> str:
        new_sequence = ''
        for v in range(1, i):
            new_sequence += step[v][ith[v] - 1]
        return new_sequence

    def next_step_stop_at(self, constraint:str, generated_sequence:str, i:int, step:Dict, ith:Dict, ith_i = 1, new = True) -> str:
        if new == True:
            next_token = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=self.trace + generated_sequence,
                logprobs = self.n,
                max_tokens = 1
            )
            i += 1
            if next_token.choices[0].logprobs.top_logprobs == []:
                return generated_sequence
            step[i] = list(next_token.choices[0].logprobs.top_logprobs[0].keys())
            ith[i] = ith_i
        passed = check_constraint(constraint, generated_sequence + step[i][ith_i - 1])
        if passed == False and step[i][ith[i] - 1] != '|<endoftext>|':
            generated_sequence += step[i][ith[i] - 1]
            return self.next_step_stop_at(constraint, generated_sequence, i, step, ith, 1, True)
        elif passed == False and step[i][ith[i] - 1] == '|<endoftext>|':
            return generated_sequence
        elif passed == True:
            generated_sequence += step[i][ith[i] - 1]
            return generated_sequence
        else:
            raise Exception('Execution error!')