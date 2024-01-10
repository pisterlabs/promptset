from dataclasses import dataclass
from typing import Callable

def load_apikey_from_env():
    """
    Load the OpenAI API key from the environment.
    """
    import os
    import openai
    from dotenv import load_dotenv
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

class DebugValues:
    verbose_logging = True

class PromptPreprocessor:
    def __init__(self, substitutions: dict[str: Callable]) -> None:
        self.substitutions = substitutions
        self.split_symbol = "##"
        return

    def preprocess_prompt(self, prompt: str) -> str:
        indexes = self.find_indexes_for_split(prompt, self.split_symbol)

        if len(indexes) % 2 != 0:  # Ensure there's an even number of split symbols
            raise ValueError("Mismatched split symbols in the prompt.")

        reconstructed_strings = []
        last_index = 0
        for i in range(0, len(indexes), 2):
            start, end = indexes[i], indexes[i+1]

            # Add the string slice before the current split symbol
            reconstructed_strings.append(prompt[last_index:start])

            # Extract the command and substitute it
            command = prompt[start + len(self.split_symbol):end]
            substitution_function = self.substitutions.get(command)
            if substitution_function and callable(substitution_function):
                substitution = substitution_function()
            else:
                substitution = command  # Default to the command if not found or not callable
            reconstructed_strings.append(substitution)

            last_index = end + len(self.split_symbol)

        # Add the remaining part of the string after the last split symbol
        reconstructed_strings.append(prompt[last_index:])

        preprocessed_string = "".join(reconstructed_strings)  # Flatten array back into string
        return preprocessed_string

    def find_indexes_for_split(self, s: str, pattern: str) -> list:
        indexes = []
        index = s.find(pattern)

        while index != -1:
            indexes.append(index)
            index = s.find(pattern, index + len(pattern))

        return indexes


if __name__ == "__main__":
    # Example usage:
    def get_hello():
        return "Hello"

    def get_world():
        return "World"

    substitutions = {
        "HELLO": get_hello,
        "WORLD": get_world
    }

    processor = PromptPreprocessor(substitutions)
    prompt = "This is a test ##HELLO## and another test ##WORLD##."
    print(processor.preprocess_prompt(prompt))  # Expected: "This is a test Hello and another test World."
