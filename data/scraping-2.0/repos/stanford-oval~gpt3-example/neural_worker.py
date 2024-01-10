"""
GPT-3 continues a prompt, works with any prompt in .txt format
"""

from typing import List
from tqdm import tqdm
import openai
from multiprocessing import Pool
from functools import partial
import math


class NeuralWorker:

    def __init__(self, prompt_template_file: str, engine: str):
        self.prompt_template: str = NeuralWorker.load_prompt_template(
            prompt_template_file)
        self.engine: str = engine

    @staticmethod
    def load_prompt_template(prompt_template_file: str) -> str:
        prompt_template = ''
        with open(prompt_template_file) as prompt_template_file:
            for line in prompt_template_file:
                if line.startswith('#'):
                    continue  # ignore comment lines in the template
                prompt_template += line
        return prompt_template

    def generate(self, input_text: str, args, postprocess=True, max_tries=1) -> str:
        """
        text-in-text-out interface to large OpenAI models
        """
        # print('input_text = ', input_text)

        # don't try multiple times if the temperature is 0, because the results will be the same
        if max_tries > 1 and args.temperature == 0:
            max_tries = 1

        # try at most `max_tries` times to get a non-empty output
        for _ in range(max_tries):
            generation_output = openai.Completion.create(engine=self.engine,
                                                         prompt=input_text,
                                                         max_tokens=args.max_tokens,
                                                         temperature=args.temperature,
                                                         top_p=args.top_p,
                                                         frequency_penalty=args.frequency_penalty,
                                                         presence_penalty=args.presence_penalty,
                                                         best_of=1,
                                                         stop=args.stop_tokens,
                                                         logprobs=0,  # log probability of top tokens
                                                         )
            # print('raw generation output = ', generation_output)
            # print('='*10)

            generation_output = generation_output['choices'][0]['text']
            generation_output = generation_output.strip()
            if postprocess:
                generation_output = self._postprocess_generations(
                    generation_output)

            if len(generation_output) > 0:
                break

        return generation_output

    def batch_generate(self, input_texts: List[str], args, postprocess=True, max_tries=1, num_processes=5) -> List[str]:
        """
        Call OpenAI's API in parallel, since each call to the biggest model takes ~1 second to return results
        """
        f = partial(self.generate, args=args,
                    postprocess=postprocess, max_tries=max_tries)
        with Pool(num_processes) as p:
            worker_outputs = list(
                tqdm(p.imap(f, input_texts), total=len(input_texts)))
        return worker_outputs

    def classify(self, input_text: str) -> float:
        """
        Binary classification interface to OpenAI models. The class labels are assumed to be ' Yes' and ' No' tokens, including the space
        Returns the probability (between 0 and 1) of the positive class (i.e. the ' Yes' label)
        """
        # print('input_text = ', input_text)
        generation_output = openai.Completion.create(engine=self.engine,
                                                     prompt=input_text,
                                                     max_tokens=1,
                                                     temperature=0,
                                                     top_p=1.0,
                                                     frequency_penalty=0,
                                                     presence_penalty=0,
                                                     best_of=1,
                                                     logprobs=10,  # returns the log probability of this many top tokens
                                                     )
        # print('raw generation output = ', generation_output)
        # print('='*10)
        logprobs = generation_output['choices'][0]['logprobs']['top_logprobs'][0]
        if ' Yes' not in logprobs and ' No' not in logprobs:
            print('Warning: the logrpob did not contain any of the classification labels.')
        pos_log = logprobs.get(' Yes', -10000)
        neg_log = logprobs.get(' No', -10000)
        return math.exp(pos_log) / (math.exp(pos_log)+math.exp(neg_log))

    def batch_classify(self, input_texts: List[str], num_processes=5) -> List[float]:
        """
        Call OpenAI's API in parallel. Is useful because each call to the biggest model takes ~1 second to return results
        """
        f = partial(self.classify)
        with Pool(num_processes) as p:
            worker_outputs = list(
                tqdm(p.imap(f, input_texts), total=len(input_texts)))
        return worker_outputs

    def _postprocess_generations(self, generation_output: str) -> str:
        """
        Might output an empty string if generation is not at least one full sentence
        """
        # replace all whitespaces with a single space
        generation_output = ' '.join(generation_output.split())

        # remove extra dialog turns, if any
        if generation_output.find('You: ') > 0:
            generation_output = generation_output[:generation_output.find(
                'You: ')]
        if generation_output.find('They: ') > 0:
            generation_output = generation_output[:generation_output.find(
                'They: ')]

        # delete half sentences
        generation_output = generation_output.strip()
        if len(generation_output) == 0:
            return generation_output

        if generation_output[-1] not in {'.', '!', '?'}:
            last_sentence_end = max(generation_output.find(
                '.'), generation_output.find('!'), generation_output.find('?'))
            if last_sentence_end > 0:
                generation_output = generation_output[:last_sentence_end+1]

        return generation_output

    def fill_prompt_template(self, **prompt_parameter_values):
        filled_prompt = self.prompt_template
        for parameter, value in prompt_parameter_values.items():
            filled_prompt = filled_prompt.replace('{'+parameter+'}', value)
        # print('filled_prompt = ', filled_prompt)
        return filled_prompt
