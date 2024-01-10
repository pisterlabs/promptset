import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import openai
import plotly.graph_objects as go
import json
import networkx as nx
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pyvis.network import Network
import random
import datetime
import re

# OpenAI API key
openai.api_key = "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"


def summarize(transcript):
    # transcript will be a pandas dataframe with columns: start, end, text
    only_text = transcript["text"].tolist()
    only_text = "\n".join(only_text)
    prompt = "".format(only_text)

    # check if cache folder exists
    if not os.path.exists("cache"):
        os.makedirs("cache")

    # save prompt to file
    tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    with open("cache/{}.txt".format(tag), "w") as f:
        f.write(prompt)

    completion = openai.Completion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    # save response to json file
    with open(f"cache/response_{tag}.json", "w") as f:
        json.dump(completion, f)

    # parse response
    summary = completion.choices[0]["text"]

    # check if summary folder exists
    if not os.path.exists("summary"):
        os.makedirs("summary")

    # save summary to file
    with open(f"summary/summary_{tag}.txt", "w") as f:
        f.write(summary)

    return summary


def elaborate(text):
    #! TODO
    prompt = ""
    pass

