import openai
from gpt3.key import get_secret_key
from colorama import Fore, Style
from gpt3.utils import query_most_likely
import random

openai.api_key = get_secret_key()

prompt = "The dog is in the house. Where is the dog?\nThe dog is"


def complete(prompt):
    ans = openai.Completion.create(prompt=prompt, n=10, engine='davinci', temperature=0)
    print(f"{prompt}{Fore.GREEN}{ans['choices'][0]['text']}{Style.RESET_ALL}")


complete(prompt)
