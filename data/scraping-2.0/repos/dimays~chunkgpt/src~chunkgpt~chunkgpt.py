"""
ChunkGPT Module

This module provides the Chunker class, which is designed to assist
users in summarizing long texts that exceed the token limit of
OpenAI's GPT-based models. It breaks down large texts into
manageable chunks, generates summaries for each chunk, and then
combines these summaries into a coherent final summary.

Usage:
from chunkgpt import Chunker

# Initialize the Chunker object
chunker = Chunker()

# Load your text to be summarized
with open('myfile.txt', 'r') as infile:
    text = infile.read()

# Generate a summary for the text
summary = chunker.summarize(text)

print(summary['result'])

Chunker Attributes:
- model (str): Name of the OpenAI GPT model.
- api_key (str): Your OpenAI API key.
- max_chunk_length (int): Maximum number of tokens for each text chunk.
- chunk_overlap (int): Number of tokens that overlap between consecutive chunks.
- summary_length (int): Maximum number of tokens for each summary.
- custom_system_msg (str): If provided, replaces ChunkGPT's default system prompt.

Chunker Methods:
- Chunker.summarize(text): Generate a summary for a given text.

Helper Methods:
- Chunker._chunk(text)
- Chunker._tokenize(text)
- Chunker._decode(tokens)
- Chunker._get_token_limit()
- Chunker._get_price_per_token()
- Chunker._get_complete_token_count(text)
- Chunker._get_base_token_count()
- Chunker._construct_messages(text)
- Chunker._get_completion(text, num_retries=3)

For more information, usage details and examples, refer to the official documentation.
GitHub repository: https://www.github.com/dimays/chunkgpt
"""

import os
import time
import openai
import tiktoken

SYSTEM_MSG = """You are SUMMIFY, a specialized AI assistant \
purpose-built to read long pieces of text and produce a concise \
summary that greatly reduces the overall length of the text \
without leaving out any critical details.

Users will hand you an entire document, or just a portion of the \
document, and it is your job to summarize the content in as few \
words as possible, while conveying the essential meaning of the text.

Your summary MUST paraphrase the document you are given so that \
it can directly replace the document 1-for-1, but with a shorter length.

Your summary MUST be concise.
Your summary MUST be comprehensive.
Your summary MUST include a concise list of ALL KEY POINTS.
Your summary MUST exclude any unnecessary fluff.
"""

USER_MSG = """
TEXT EXCERPT:
{}

SUMMARY:
"""

class Chunker:
    def __init__(self, 
                 api_key: str = os.getenv('OPENAI_API_KEY'),
                 model: str = 'gpt-3.5-turbo',
                 max_chunk_length: int = 2048,
                 chunk_overlap: int = 50,
                 summary_length: int = 1024,
                 temperature: float = 0.0,
                 custom_system_msg: str = None
                 ):
        if not isinstance(api_key, str):
            raise TypeError(f"invalid api_key '{api_key}'; must be a string")
        
        try:
            openai.api_key = api_key
            models = [model['id'] for model in openai.Model.list()['data']]
        except:
            raise ValueError(f"Unable to access OpenAI's GPT model list; ensure your API key ({api_key}) is correct")

        if not isinstance(model, str):
            raise TypeError(f"invalid model ({model}); must be a string")
        
        if model not in models:
            raise ValueError(f"invalid model ({model}); model not found in the list of available OpenAI models")
        
        if not isinstance(max_chunk_length, int):
            raise TypeError(f"invalid max_chunk_length ({max_chunk_length}); must be an integer")

        if max_chunk_length <= 0:
            raise ValueError(f"invalid max_chunk_length ({max_chunk_length}); must be an integer greater than 0")

        if not isinstance(chunk_overlap, int):
            raise TypeError(f"invalid chunk_overlap ({chunk_overlap}); must be an integer")

        if chunk_overlap < 0:
            raise ValueError(f"invalid chunk_overlap ({chunk_overlap}); must be an integer greater than or equal to 0")

        if chunk_overlap > max_chunk_length:
            raise ValueError(f"invalid chunk_overlap ({chunk_overlap}); must be smaller than max_chunk_length ({max_chunk_length})")

        if not isinstance(summary_length, int):
            raise TypeError(f"invalid summary_length ({summary_length}); must be an integer")
        
        if summary_length <= 0:
            raise ValueError(f"invalid summary_length ({summary_length}); must be an integer greater than 0")

        if not isinstance(temperature, float):
            raise TypeError(f"invalid temperature ({temperature}); must be a float")

        if temperature < 0 or temperature > 2:
            raise ValueError(f"invalid temperature ({temperature}); must be a float bewteen 0 and 2 inclusive")

        self.model = model
        self.api_key = api_key
        self.max_chunk_length = max_chunk_length
        self.chunk_overlap = chunk_overlap
        self.summary_length = summary_length
        self.temperature = temperature

        if custom_system_msg:
            self.system_msg = custom_system_msg
        else:
            self.system_msg = SYSTEM_MSG

        self.encoding = tiktoken.encoding_for_model(model)
        self.token_limit = self._get_token_limit()
        self.price_per_token = self._get_price_per_token()

        allowed_tokens = self._get_base_token_count() + summary_length + max_chunk_length
        if allowed_tokens > self.token_limit:
            exceeded_by = allowed_tokens - self.token_limit
            raise ValueError(f"Combined summary_length and max_chunk_length exceeds {self.model}'s token_limit by {exceeded_by} tokens. Reduce one or more of these parameters or switch models.")

        print("Initialized Chunker.")

    def summarize(self, text, final_step='summarize'):
        """Generate a summary for a text and some metadata about the summarizing process.
        
        final_step: 
          - if 'summarize' (default), the final step will be to summarize the combined intermediate summaries.
          - if 'combine', the final step will be to combine the intermediate summaries.
        """
        # Initialize summary dict
        summary = {
            'original': text,
            'result': '',
            'chunks': {},
            'intermediate_steps': [],
            'total_tokens': 0,
            'cost': 0.0
        }

        # Split text into chunks
        chunks = self._chunk(text)
        if len(chunks) > 1:
            print(f"Split text into {len(chunks)} chunks.")

        # Initialize combined summary string
        combined_summaries = ""

        # Get a completion for each chunk, update dict with details
        for i, chunk in enumerate(chunks):
            completion = self._get_completion(chunk)
            content = completion['choices'][0]['message']['content']
            combined_summaries += f"{content}\n"
            summary['chunks'][i+1] = {
                'input': chunk,
                'output': content
            }
            total_tokens = completion['usage']['total_tokens']
            summary['total_tokens'] += total_tokens
            summary['cost'] += total_tokens * self.price_per_token
            step = f"Got completion for chunk {i+1}."
            summary['intermediate_steps'].append(step)
            print(step)

        # Calculate token count of combined summary string
        combined_token_cnt = self._get_complete_token_count(combined_summaries)

        # Simply return combined summaries if final step is 'combine'.
        if final_step == 'combine':
            content = combined_summaries
            summary['result'] = content
            step = "Returned combined summaries as final step."
            summary['intermediate_steps'].append(step)
            print(step)
        # Otherwise, if 'summarize', reduce size of combined summary string if necessary
        elif final_step == 'summarize' and combined_token_cnt >= self.token_limit:
            reduced_summary = self.summarize(combined_summaries)
            new_token_cnt = self._get_complete_token_count(reduced_summary['result'])
            summary['total_tokens'] += reduced_summary['total_tokens']
            summary['cost'] += reduced_summary['cost']
            step = f"Reduced summary from {combined_token_cnt} to {new_token_cnt}."
            summary['intermediate_steps'].append(step)
            summary['result'] = reduced_summary['result']
            print(step)
        # Otherwise get summary for combined summaries.
        elif final_step == 'summarize':
            completion = self._get_completion(combined_summaries)
            content = completion['choices'][0]['message']['content']
            summary['result'] = content
            total_tokens = completion['usage']['total_tokens']
            summary['total_tokens'] += total_tokens
            summary['cost'] += total_tokens * self.price_per_token
            step = "Got completion for combined summaries."
            summary['intermediate_steps'].append(step)
            print(step)
        else:
            raise ValueError(f"'final_step' option {final_step} unrecognized. Expects 'summarize' or 'combine'")

        return summary

    def _chunk(self, text):
        """Returns list of strings, each string representing a chunk
        of the original text that needs to be summarized. If the 
        original text plus the system message and expected output
        is within the token limit, returns a single-item list containing
        the input text."""
        total_cnt = self._get_complete_token_count(text)
        if total_cnt < self.token_limit:
            return [text]

        tokens = self._tokenize(text)
        chunks = []

        chunk_step = self.max_chunk_length - self.chunk_overlap
        for i in range(0, len(tokens), chunk_step):
            start = i
            end = i + self.max_chunk_length
            chunk_tokens = tokens[start:end]
            chunk = self._decode(chunk_tokens)
            chunks.append(chunk)
        return chunks

    def _tokenize(self, text):
        """Returns an encoded list of tokens from the given text string."""
        return self.encoding.encode(text)
    
    def _decode(self, tokens):
        """Returns a decoded string from a list of tokens."""
        return self.encoding.decode(tokens)
    
    def _get_token_limit(self):
        """Returns the token limit for each gpt-3.5 and gpt-4 model,
        current as of August 13, 2023."""
        if self.model.startswith('gpt-3.5-turbo-16k'):
            token_limit = 16384
        elif self.model.startswith('gpt-4-32k'):
            token_limit = 32768
        elif self.model.startswith('gpt-4'):
            token_limit = 8192
        else:
            token_limit = 4096
        return token_limit

    def _get_price_per_token(self):
        """Returns the price per token in cents for each gpt-3.5 and
        gpt-4 model, current as of August 13, 2023."""
        if self.model.startswith('gpt-3.5-turbo-16k'):
            price = 0.0003
        elif self.model.startswith('gpt-4-32k'):
            price = 0.006
        elif self.model.startswith('gpt-4'):
            price = 0.003 
        else:
            price = 0.00015
        return price

    def _get_complete_token_count(self, text):
        """Returns total number of tokens for a given input text,
        including the system prompt and expected number of tokens needed
        for the completion output."""
        system_msg_token_cnt = len(self._tokenize(self.system_msg))
        user_msg_token_cnt = len(self._tokenize(USER_MSG.format(text)))
        summary_token_cnt = self.summary_length
        total = system_msg_token_cnt + user_msg_token_cnt + summary_token_cnt
        return total

    def _get_base_token_count(self):
        """Returns total number of tokens used by the system message and
        user message (prior to formatting with the input text)."""
        system_msg_token_cnt = len(self._tokenize(self.system_msg))
        user_msg_token_cnt = len(self._tokenize(USER_MSG.format("")))
        total = system_msg_token_cnt + user_msg_token_cnt
        return total

    def _construct_messages(self, text):
        """Constructs a list of messages to submit to OpenAI for
        chat completion."""
        messages = [
            {
                'role': 'system',
                'content': self.system_msg
            },
            {
                'role': 'user',
                'content': USER_MSG.format(text)
            }
        ]
        return messages

    def _get_completion(self, text, num_retries=3):
        """Uses the designated system_msg and the default USER_MSG template
        to get an OpenAI chat completion for the provided text.
        
        If the completion fails, ChunkGPT will retry up to 3
        times, with a delay of 1 second between retries."""
        msgs = self._construct_messages(text)
        try:
            completion = openai.ChatCompletion.create(
                model=self.model,
                temperature=self.temperature,
                messages=msgs,
                max_tokens=self.summary_length
            )
        except Exception as e:
            print(f"Completion failed: {e}")
            if num_retries > 0:
                print("Retrying...")
                time.sleep(1)
                return self._get_completion(text, num_retries-1)
            else:
                print("No more retries.")
                raise ValueError("Unable to return completion.")
        return completion
