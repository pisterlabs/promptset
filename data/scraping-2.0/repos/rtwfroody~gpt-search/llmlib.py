"""
Library that provides basic function for using large language models (LLMs).

Currently only supports OpenAI.
"""

import os
import re
import textwrap

from diskcache import Cache
import appdirs
import openai
import tiktoken

def split_separator(text, separator):
    """Split a text using a separator, but keep the separator in the result.

    Separator must be a regex with two capture groups. The first one is kept
    with the text before the split, the second one is kept with the text after
    the split."""
    parts = []
    remainder = text
    before_remainder = ""
    while remainder:
        split = re.split(separator, remainder, maxsplit=1, flags=re.MULTILINE)
        if len(split) == 1:
            parts.append(before_remainder + remainder)
            break
        part, after_part, next_before_remainder, next_remainder = split
        parts.append(before_remainder + part + after_part)
        remainder = next_remainder
        before_remainder = next_before_remainder
    return parts

def quote(text, prefix='> '):
    """Quote a text, preserving paragraphs and line breaks."""
    paragraphs = text.splitlines()
    wrapped_paragraphs = [textwrap.wrap(p) for p in paragraphs]
    lines = "\n".join("\n".join(p) for p in wrapped_paragraphs)
    quoted_lines = re.sub(r"^", prefix, lines, flags=re.MULTILINE)
    return quoted_lines

class Api:
    """Abstract base class for APIs to LLMs."""
    def ask(self, prompt):
        """Ask the model a question."""
        raise NotImplementedError

    def token_count(self, prompt):
        """Return the number of tokens in the prompt."""
        raise NotImplementedError

    def max_token_count(self):
        """Return the maximum number of tokens that can be sent to the model."""
        raise NotImplementedError

class Openai(Api):
    """API to OpenAI's GPT model."""
    def __init__(self, model="gpt-3.5-turbo", verbose=False, api_key=None):
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.verbose = verbose

    def ask(self, prompt):
        """Ask the model a question."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0]['message']['content']
        except openai.error.InvalidRequestError as exception:
            exception._message += f"; computed token length={self.token_count(prompt)}"
            raise
        return result

    def token_count(self, prompt):
        """Return the number of tokens in the prompt."""
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(prompt))

    def max_token_count(self):
        """Return the maximum number of tokens that can be sent to the model."""
        # I think this compensates for the overhead in the messages dict.
        overhead_tokens = 8
        return {
                "gpt-4": 8192,
                "gpt-3.5-turbo": 4097
            }.get(self.model, 4096) - overhead_tokens

    def __repr__(self) -> str:
        return f"Openai({self.model})"

class Llm:
    """Interface to a large language model (LLM)."""
    def __init__(self, api : Api, verbose=False):
        self.api = api
        self.verbose = verbose
        self.cache = Cache(appdirs.user_cache_dir("llmlib"))
        self.counters = {}
        log_dir = appdirs.user_log_dir("llmlib")
        log_path = os.path.join(log_dir, "log.txt")
        if self.verbose:
            print(f"Logging to {log_path}")
        os.makedirs(log_dir, exist_ok=True)
        # pylint: disable-msg=consider-using-with
        self.log_fd = open(log_path, "a", encoding="utf-8")

    def _log(self, text : str):
        """Log text to the log file."""
        self.log_fd.write(text)
        if not text.endswith("\n"):
            self.log_fd.write("\n")

    def ask(self, prompt : str):
        """Ask the model a question."""
        self._log(f"\nAsk {self.api!r}:\n{quote(prompt)}")
        if self.verbose:
            print(f"Ask {self.api!r}: {prompt[:60]!r}")

        assert len(prompt) > 25

        cache_key = ("ask", repr(self.api), prompt)
        result = self.cache.get(cache_key)
        self._increment_counter(f"ask {self.api!r}")

        if result:
            self._increment_counter(f"ask-hit {self.api!r}")
            cached = " (cached)"
        else:
            self._increment_counter(f"ask-miss {self.api!r}")
            result = self.api.ask(prompt)
            cached = ""

        self._log(f"\nResponse{cached}:\n{quote(result)}")
        if self.verbose:
            print(f"Response{cached}: {result[:60]!r}")

        self.cache[cache_key] = result

        return result

    def _increment_counter(self, name):
        """Increment a counter."""
        self.counters.setdefault(name, 0)
        self.counters[name] += 1

    def split_markdown(self, text, token_limit=None):
        """Split a markdown text to fit the given token limit."""
        return self.split_text(text, token_limit=token_limit,
                               separators=(
                                   r"()(^# .*$)",
                                   r"()(^## .*$)",
                                   r"()(^### .*$)",
                                   r"()(^#### .*$)",
                                   r"(\n(?:\s*\n)+)",
                                   r"(\n+)",
                                   r"(\s+)"))

    def split_text(self, text, token_limit=None,
                   separators=(r"(\n(?:\s*\n)+)()", r"(\n+)()", r"(\s+)()")):
        """Split a text into parts which each fit the given token limit."""
        if token_limit is None:
            token_limit = self.api.max_token_count()

        # Split text into parts that are each short enough to fit the token limit.
        short_parts = []
        for part in split_separator(text, separators[0]):
            if self.api.token_count(part) > token_limit:
                short_parts.extend(self.split_text(part, token_limit, separators[1:]))
            else:
                short_parts.append(part)

        # Combine short parts into longer ones that still fit the token limit.
        parts = []
        for part in short_parts:
            if parts and self.api.token_count(parts[-1] + part) <= token_limit:
                parts[-1] += part
            else:
                parts.append(part)
        return parts

    def summarize(self, text, token_limit=None, prompt="Summarize:", max_iterations=10):
        """Summarize a text to fit the given token limit."""
        max_tokens = self.api.max_token_count() - self.api.token_count(prompt)
        if token_limit is None:
            token_limit = max_tokens
        else:
            token_limit = min(token_limit, max_tokens)
        for _ in range(max_iterations):
            if self.api.token_count(text) <= token_limit:
                break
            text = "\n\n".join(
                self.ask(f"{prompt} {part}")
                for part in self.split_text(text, token_limit=token_limit))
        return text

    def counter_string(self, pattern="^ask "):
        """Return a string representation of the counters."""
        return "; ".join(
            f"{name}:{count}"
            for name, count in self.counters.items()
            if re.search(pattern, name))

    def get_num_tokens(self, text):
        """Return the number of tokens in the text."""
        return self.api.token_count(text)
