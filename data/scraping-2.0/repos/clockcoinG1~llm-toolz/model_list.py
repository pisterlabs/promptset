import datetime
import json
import math
import os
import sys
from random import random

import openai

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)


def has_colours(stream):
    if not hasattr(stream, "isatty"):
        return False

    if not stream.isatty():
        return False
    return True


def printout(text, colour=CYAN):
    if has_colours(sys.stdout):
        seq = "\x1b[1;%dm" % (30 + colour) + text + "\x1b[0m"
        sys.stdout.write(seq)
        return seq
    else:
        sys.stdout.write(text)
        return "\033[1;" + str(colour) + "m" + text + "\033[0m"


def main():
    my_models = openai.Model.list()
    model_names = [[model["id"], model["created"]] for model in my_models["data"]]

    for model in model_names:
        model[1] = datetime.datetime.fromtimestamp(model[1]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    model_names.sort(key=lambda x: x[1])
    model_names.reverse()
    for i, model in enumerate(model_names):
        if len(model[0]) < 40:
            model[0] = model[0] + " " * (40 - len(model[0]))
        printout(str(i) + " " + model[0] + " " + model[1] + "\n", int(random() * 100))


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        printout(
            "Please set the environment variable OPENAI_API_KEY to your API key.\nPendejo mofo",
            int(random() * 100),
        )
        sys.exit(1)
    main()
