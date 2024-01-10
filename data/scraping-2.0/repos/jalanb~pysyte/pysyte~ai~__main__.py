"""Example usage of OpenAPI calls"""

import sys

from rich import print

from pysyte import os
import pysyte.ai.open import OpenaiApp
import pysyte.ai.open import wwts
from pysyte.oss.getch import ask_user_simplified
from pysyte.types.dictionaries import NameSpaces


models = NameSpaces(
    dict(
        davinci="text-davinci-003",
        curie="text-curie-001",
        babbage="text-babbage-001",
        ada="text-ada-001",
    )
)



def main():

    app = OpenaiApp("wwts")
    app = pysyte.ai.open.OpenaiApp()

    breakpoint()
    question = " ".join(sys.argv[1:]) or config.prompt.final
    messages=[
        {"role": "system", "content": config.prompt.prefix},
        {"role": "system", "content": config.prompt.context},
        {"role": "system", "content": config.prompt.task},
        {"role": "system", "content": config.prompt.rules},
        {"role": "user", "content": question},
    ]
    choices = app.ask([
    ], config)

    allowed = []
    for i, choice in enumerate(choices, 1):
        reply = choice['message']['content']
        print(f"Reply {i}: {reply}\n\n")
        allowed.append(str(i))
    answer = ask_user_simplified("Which reply is best?", "")
    if answer not in allowed:
        return False
    choice = choices[int(answer) - 1]
    messages.append({"role": "assistant", "content": choice['message']['content']})
    question = input("Ask more? ")
    if not question:
        return True
    messages.append({"role": "user", "content": question})
    choices = app.ask(messages, config)
    for i, choice in enumerate(choices, 1):
        reply = choice['message']['content']
        print(f"Reply {i}: {reply}\n\n")


if __name__ == "__main__":
    x = os.EX_OK if main() else 1
    sys.exit(x)

