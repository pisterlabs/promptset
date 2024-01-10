from openai import OpenAI
import tiktoken
import warnings
from typing import List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

client = OpenAI()

# See https://platform.openai.com/docs/guides/rate-limits/error-mitigation?context=tier-free
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

# For more info on available models, see https://platform.openai.com/docs/models/overview
MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k"
  ]

def get_default_max_input_tokens(model_name: str) -> int:
    """Returns the default maximum number of tokens that can be sent to the model, before running intro troubles."""
    max_tokens = None

    if model_name == "gpt-3.5-turbo":
        max_tokens = 4096
    elif model_name == "gpt-3.5-turbo-16k":
        max_tokens = 16384
    elif model_name == "gpt-4":
        max_tokens = 8192
    elif model_name == "gpt-4-32k":
        max_tokens = 32768
    else:
        raise ValueError(f"Unknown model name '{model_name}'.")

    # 0.31 is a magic number that allows us enough space to add the system context and the translation prompt, while also taking into account the output of the model.
    max_input_tokens = int(max_tokens * 0.31)
    return max_input_tokens

class GPTTranslator:
    """A class for translating LaTex text using OpenAI's API."""

    def __init__(self, lang_from:str = "English", 
                 lang_to:str = "Spanish", 
                 model_name: str = "gpt-3.5-turbo-16k",
                 ignore_commented_blocks: bool = True,
                 verbose: bool = False,
                 max_input_tokens: int = None):

        self.model_name = model_name
        self.system_context = f"You are a helpful assistant that translates {lang_from} to {lang_to}."
        self.translation_prompt = f"Translate the following LaTex text from {lang_from} to {lang_to}. "

        if max_input_tokens is None:
            max_input_tokens = get_default_max_input_tokens(model_name)
            if verbose:
                print(f"Using default max_input_tokens: {max_input_tokens}")

        translation_prompt_tokens = self._count_tokens(self.translation_prompt)
        if verbose:
            print(f"Translation prompt tokens: {translation_prompt_tokens}")

        system_context_tokens = self._count_tokens(self.system_context)
        if verbose:
            print(f"System context tokens: {system_context_tokens}")

        self.max_text_tokens = max_input_tokens - translation_prompt_tokens - system_context_tokens
        if verbose:
            print(f"Max text tokens: {self.max_text_tokens}")

        self.ignore_commented_blocks = ignore_commented_blocks
        self.verbose = verbose

    def compute_translation_tokens(self, string: str) -> int:
        """Returns the total number of tokens that would be sent to the model."""

        if self.ignore_commented_blocks:
            blocks = self._separate_commented_blocks(string)
            return sum([self._compute_translation_tokens_in_block(block) 
                        for block in blocks if not block.startswith("%")])
        else:
            return self._compute_translation_tokens_in_block(string)

    def _compute_translation_tokens_in_block(self, block: str) -> int:
        """Returns the number of tokens that would be sent to the model for a single block."""

        chunks = self._split_into_max_length_chunks(block)
        res = self._count_tokens(self.translation_prompt) + self._count_tokens(self.system_context)
        for s in chunks:
            res += self._count_tokens(s)

        return res

    def _count_tokens(self, string: str) -> int:
        """Returns the number of tokens in a text string."""

        encoding = tiktoken.encoding_for_model(self.model_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def translate(self, string: str) -> str:

        if self.ignore_commented_blocks:
            return self._translate_uncommented_blocks(string)
        else:
            return self._translate_block(string)

    def _translate_uncommented_blocks(self, string: str) -> str:
        """Translate a piece of LaTex text using OpenAI's API. Only translate non-commented blocks."""

        blocks = self._separate_commented_blocks(string)

        res = []
        for block in blocks:
            if block.startswith("%") or block.strip() == "":
                res.append(block)
            else:
                res.append(self._translate_block(block))

        return "\n".join(res)

    def _separate_commented_blocks(self, string: str) -> List[str]:
        """Split a string into blocks. Each block is a sequence of lines where either 
        all of them are commented out or non of them are."""

        blocks = []
        block = ""
        commented_block = False
        for line in string.splitlines():
            if line.startswith("%"):
                if not commented_block:
                    blocks.append(block)
                    block = line
                    commented_block = True
                else: 
                    block += "\n" + line 
            else:
                if commented_block:
                    blocks.append(block)
                    block = line
                    commented_block = False
                else: 
                    block += "\n" + line 
        blocks.append(block)

        return blocks

    def _translate_block(self, string: str) -> str:
        """Translate a piece of LaTex text using OpenAI's API."""

        chunks = self._split_into_max_length_chunks(string)
        res = []
        for s in chunks:
            res.append(self._translate_chunk(s))

        return "\n".join(res)

    def _split_into_max_length_chunks(self, string: str) -> List[str]:
        """Split a string into chunks of max_text_tokens length. Only split on newlines."""

        chunks = []
        chunk = None
        for line in string.splitlines():
            updated_chunk = chunk + "\n" + line if chunk != None else line
            if self._count_tokens(updated_chunk) <= self.max_text_tokens:
                chunk = updated_chunk
            else:
                chunks.append(chunk)
                chunk = line
        
        if chunk != None:
            chunks.append(chunk)

        return chunks

    def _translate_chunk(self, string: str) -> str:
        """Perform call to OpenAI's API to translate a chunk of text."""

        messages = [
                {"role": "system", "content": self.system_context},
                {"role": "user", "content": self.translation_prompt + " "+ string}
            ]
        completion = completion_with_backoff(
            model=self.model_name,
            messages=messages
        )

        if completion.choices[0].finish_reason == "length":
            warnings.warn("Incomplete model output due to max_tokens parameter or token limit.")

        if self.verbose:
            print("")
            print("------------------ MESSAGES (API's input) ------------------")
            print(messages)
            print("")
            print("------------------ COMPLETION (API's output) ------------------")
            print(completion)
            print("")

        return completion.choices[0].message.content
 