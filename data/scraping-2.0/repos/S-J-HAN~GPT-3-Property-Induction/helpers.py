import config

import pandas as pd

import openai


def read_osherson_ranked_arguments(conclusion_type: str):
    if conclusion_type == "specific":
        columns = ["premise_1", "premise_2", "conclusion", "human_strength"]
    else:
        columns = ["premise_1", "premise_2", "premise_3", "human_strength"]
    df = pd.read_csv(config.ranked_arguments_path(conclusion_type), sep="   ", names=columns, engine="python")
    for col in columns[:-1]:
        df[col] = df[col].apply(int).apply(lambda x: config.PREMISE_NUMBERS[x])
    if conclusion_type == "general":
        df["conclusion"] = ["all mammals"]*df.shape[0]
    else:
        df["premise_3"] = [None]*df.shape[0]
    
    return df


def get_embedding(text, engine="davinci-similarity"):
    text = text.replace("\n", " ")
    return openai.Engine(id=engine).embeddings(input = [text])['data'][0]['embedding']
