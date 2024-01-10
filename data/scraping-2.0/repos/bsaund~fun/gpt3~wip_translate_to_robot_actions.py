import openai
from gpt3.key import get_secret_key
from colorama import Fore, Style

from gpt3.utils import query_most_likely

openai.api_key = get_secret_key(filename="/home/bsaund/.openai_key")

actions = ["pick", "place", "move", "label"]
objects = ["bowl", "plate", "fork", "cup"]


prime_text = """Give the robot explicit commands.
###
Grab the bowl => pick bowl
###
Drop the silverware => place fork
###
Clean up the soup => pick bowl
###
Deliver the water => place cup
###
"""


def command(query):
    prompt = prime_text + query + " =>"
    # response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=1, logprobs=100, n=1)
    action = query_most_likely(prompt, actions, print_probs=True)
    prompt += " " + action
    object = query_most_likely(prompt, objects, print_probs=True)

    print(f'{Fore.GREEN}{query}: {Style.RESET_ALL}', end='')
    print(f'{Fore.MAGENTA}{action} {object}{Style.RESET_ALL}')


command("shift the utensils around")
command("Bring me the steak")
command("Set the table")
command("label this location")
command("drink the water")
command("Drink the cool aid")
command("Place spoon")
command("take the utensil from the table")
