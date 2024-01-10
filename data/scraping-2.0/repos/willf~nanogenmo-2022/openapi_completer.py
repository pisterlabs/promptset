import configparser
import itertools
import json
import os
import re
import sys
import textwrap
import time

import openai
from nltk import sent_tokenize

# Not much is left, but I was grateful for the inspiration from this code:
# https://github.com/daveshap/RecursiveSummarizer/blob/main/recursively_summarize.py

def flatten(list_of_lists):
    return itertools.chain.from_iterable(list_of_lists)

def combine_two_chunks_if_shorter_than_maxsize(chunk1, chunk2, maxsize):
    if len(chunk1) + len(chunk2) + 1 <= maxsize:
        return chunk1 + " " + chunk2
    return None

def head(iterable, n=1):
    return next(itertools.islice(iterable, n))

def tail(iterable, n=1):
    return itertools.islice(iterable, n, None)

def resize_chunks(chunks, maxsize):
    """
    We have a bunch of chunks that we might be able to combine into larger chunks of maxsize.
    But the sub-chunks have to be joined by spaces, so we need to make sure that we don't
    exceed the maxsize.
    >>> list(resize_chunks(["a", "b", "c", "d", "e", "f", "g"], 5))
    ["a b c", "d e f", "g"]
    """
    if not chunks:
        return chunks
    resized = []
    current_work = head(chunks, 1)
    for chunk in tail(chunks, 1):
        combined = combine_two_chunks_if_shorter_than_maxsize(current_work, chunk, maxsize)
        if combined:
            current_work = combined
        else:
            resized.append(current_work)
            current_work = chunk
    resized.append(current_work)
    return resized

def replace_newlines_with_spaces(text):
    return re.sub(r"\n", " ", text)

def break_into_paragraphs(text):
    return text.split("\n\n")

class ConfigReader():
    def __init__(self, config_var, config_file='.env', default=None):
        self.config_var = config_var
        self.default = default
        self.config_file = config_file

    def read(self):
        """
        Reads an config variable.
        It tries the following:
        1. If the config variable is set, it returns the value.
        2. If the config variable is not set, it tries to read it from a Config file.
        3. If the config variable is not set and the Config file is not found, it used the default value.
        4. Otherwise, it raises an exception.

        """
        value = os.getenv(self.config_var)
        if value is not None:
            return value
        if self.config_file:
            path = os.path.join(os.path.dirname(__file__), self.config_file)
            config = configparser.ConfigParser()
            config.read(path)
            value = config['DEFAULT'][self.config_var]
            if value is not None:
                return value
        if self.default is not None:
            return self.default
        raise Exception(f"Config variable {self.config_var} is not set.")

class ChunkingCompleter():
    def __init__(self, prompt_file='prompt.template',
        variable_name='{{TEXT}}',
        engine='text-davinci-002',
        temperature=0.6,
        top_p=1.0,
        frequency_penalty=0.25,
        presence_penalty=0.0,
        stop=['<<END>>'],
        max_tokens=3750,
        chunk_size=1000,
        max_tries=3):
        openai.api_key = ConfigReader('OPENAI_API_KEY').read()
        self.prompt_file = prompt_file
        self.variable_name = variable_name
        self.engine = engine
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.max_tries = max_tries

    def get_prompt(self):
        """Gets the prompt template."""
        path = os.path.join(os.path.dirname(__file__), self.prompt_file)
        with open(path) as f:
            prompt = f.read()
        return prompt

    def chunk_text(self, text):
        """Chunks text into chunks of size chunk_size."""
        text = replace_newlines_with_spaces(text).strip()
        sentences = sent_tokenize(text)
        chunked_sentences = [textwrap.wrap(sentence, self.chunk_size) for sentence in sentences]
        flattened = flatten(chunked_sentences)
        chunks = [chunk for chunk in flattened if len(chunk) > 0]
        chunks = resize_chunks(chunks, self.chunk_size)
        return chunks

    def complete_chunk(self, chunk, prompt):
        """Completes a chunk of text."""
        tries = 1
        while tries < self.max_tries:
            tries += 1
            try:
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt.replace(self.variable_name, chunk),
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=self.stop,
                    max_tokens=self.max_tokens
                )

                completion_chunk = response["choices"][0]["text"].strip()
                completion_chunk = re.sub('\s+', ' ', completion_chunk)
                return {'completion': completion_chunk, 'success': True}
            except Exception as e:
                print(e)
                sys.stderr.write('Error: {{}} Retrying... in {{}} seconds.\n'.format(e, tries-1))
                sys.stderr.flush()
                time.sleep(tries-1)
        return {'completion': '', 'success': False}

    def complete(self, text):
        """Completes text."""
        prompt = self.get_prompt()
        chunks = self.chunk_text(text)
        for chunk in chunks:
            completion = self.complete_chunk(chunk, prompt)
            result = {
                'chunk': chunk,
                'completion': completion["completion"],
                'success': completion["success"]
            }
            yield result

if __name__ == '__main__':
    completer = ChunkingCompleter()
    text = sys.stdin.read()
    paragraphs = break_into_paragraphs(text)
    for paragraph in paragraphs:
        conversion = ' '.join([result['completion'] for result in completer.complete(paragraph)])
        result = {'paragraph': paragraph, 'conversion': conversion}
        sys.stdout.write(json.dumps(result))
        sys.stdout.write("\n")
        sys.stdout.flush()
