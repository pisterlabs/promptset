"""
Generates the prompts and collects the model's responses.

2-shot 2-digit positive integer multiplication evaluated on GPT-3.
We always list the larger number first.
"""

import os
import time
import csv
from abc import ABC, abstractmethod

from tqdm import tqdm
import openai
from openai.error import RateLimitError
import backoff
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

START_MULTIPLICAND = 0
STOP_MULTIPLICAND = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

# See ./generate_prompt.py
FEW_SHOT_PROMPT = 'Multiply:\n41 * 70 = 2870\n23 * 6 = 138\n9 * 58 = 522\n'


def get_output_filename(model, suffix='') -> str:
    """Return a filename for the output data."""
    if suffix != '':
        suffix = f'_{suffix}'
    return f'./data/multiplication_data_{model}_{START_MULTIPLICAND}_{STOP_MULTIPLICAND}{suffix}.csv'


class Prompter(ABC):
    """Given numbers to multiply, format them into a prompt."""

    @abstractmethod
    def __call__(self, a: int, b: int) -> str:
        raise NotImplementedError


class FewShotPrompter(Prompter):
    """2-shot example prompting to help our model learn the format."""

    def __init__(self, examples: str = ''):
        if examples == '':
            self.examples = 'Multiply:\n7 * 6 = 42\n65 * 44 = 2860\n98 * 23 = 2254\n'  # Randomly chosen 2-shot default
        else:
            self.examples = examples

    def __call__(self, a: int, b: int) -> str:
        return f'{self.examples}{str(a)} * {str(b)} ='


class Answerer(ABC):
    """Answers multiplication prompts."""

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, prompts: list[str]) -> list[str | None]:
        """
        Answer a list of prompt strings, returning a list of answers to each prompt as ints.

        Returns None if no valid answer could be extracted for a given prompt.
        """
        raise NotImplementedError


class StubAnswerer(Answerer):
    """Output canned answers for testing."""

    def __repr__(self) -> str:
        return 'StubAnswerer'

    def __call__(self, prompts: list[str]) -> list[str]:
        return ['66'] * len(prompts)


class GPT3APIAnswerer(Answerer):
    def __init__(self, model_name):
        self.model_name = model_name

    def __repr__(self) -> str:
        return f'GPT3-{self.model_name}'

    @backoff.on_exception(backoff.expo, RateLimitError)
    def completions_with_backoff(self, **kwargs):
        """https://help.openai.com/en/articles/5955604-how-can-i-solve-429-too-many-requests-errors"""
        response = openai.Completion.create(**kwargs)
        return response

    def __call__(self, prompts: list[str]) -> list[str | None]:
        openai.api_key = os.getenv("OPENAI_API_KEY")

        output = []
        try:
            for i in tqdm(range(0, len(prompts), 20)):
                batch = prompts[i:i+20]
                completion = self.completions_with_backoff(
                    model=self.model_name,
                    prompt=batch,
                    max_tokens=6,
                    temperature=0,
                    stop='\n'
                )
                output += [choice.text.strip() for choice in completion['choices']]  # type: ignore
        except Exception as e:
            print(e)
        return output


class HFTransformersAnswerer(Answerer):
    def __init__(self, model_name, batch_size=128):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.eos_token = self.tokenizer('\n').input_ids[0]
        self.tokenizer.pad_token = self.eos_token

    def __repr__(self) -> str:
        # Convert / to - to avoid creating a new directory
        return f'HF-{self.model_name.replace("/", "-")}'

    def __call__(self, prompts: list[str]) -> list[str | None]:
        openai.api_key = os.getenv('OPENAI_API_KEY')

        output = []
        for i in tqdm(range(0, len(prompts), self.batch_size)):
            batch = prompts[i:i+self.batch_size]
            tokenized = self.tokenizer(batch, padding=True, return_tensors='pt')
            input_ids = tokenized.input_ids.to(DEVICE)
            attention_mask = tokenized.attention_mask.to(DEVICE)
            generated = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                temperature=0.0,
                pad_token_id=self.eos_token,
                eos_token_id=self.eos_token,
            )
            only_new_tokens = generated[:, tokenized.input_ids[0].shape[0]:]
            decoded_completions = self.tokenizer.batch_decode(only_new_tokens)
            output += [completion.strip() for completion in decoded_completions]
        return output


if __name__ == '__main__':
    # Generate numbers and prompts
    multiplicand_tuples = []
    prompts = []
    prompter = FewShotPrompter(FEW_SHOT_PROMPT)

    print('Generating prompts...')
    for a in range(START_MULTIPLICAND, STOP_MULTIPLICAND):
        # for b in range(START_MULTIPLICAND, a + 1):  # a >= b
        for b in range(0, STOP_MULTIPLICAND):  # a not necessarily >= b
            multiplicand_tuples.append((a, b))
            prompts.append(prompter(a, b))

    # Choose a model with which to answer
    for answerer in [
        StubAnswerer(),  # For testing, always outputs a constant
        HFTransformersAnswerer('gpt2'),  # 117M
        HFTransformersAnswerer('EleutherAI/gpt-neo-1.3B', batch_size=16),  # 1.3B
        # HFTransformersAnswerer('EleutherAI/gpt-j-6B'),  # 6B, not enough memory on my machine
        # See https://blog.eleuther.ai/gpt3-model-sizes/ for curie size estimate
        # GPT3APIAnswerer('text-ada-001'),  # ~350M
        # GPT3APIAnswerer('text-babbage-001'),  # ~1.3B
        # GPT3APIAnswerer('text-curie-001'),  # ~6.7B
        # GPT3APIAnswerer('text-davinci-003'),  # 175B
    ]:
        # Generate some answers
        print(f'Generating answers with model {answerer}')
        start_time = time.time()
        answers = answerer(prompts)
        duration = time.time() - start_time

        # Print the results for debugging
        # for multiplicand_tuple, answer in zip(multiplicand_tuples, answers):
        #     a, b = multiplicand_tuple
        #     print(f'{a} * {b} = {answer}')

        num_answered = len([answer for answer in answers if answer is not None])
        print(f'{answerer}: Generated {len(prompts)} prompts and successfully got {num_answered} answers in {duration} seconds.')

        # Write results as CSV
        with open(get_output_filename(answerer), 'w', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow(['a', 'b', 'completion'])
            for multiplicand_tuple, answer in zip(multiplicand_tuples, answers):
                writer.writerow([multiplicand_tuple[0], multiplicand_tuple[1], answer])
