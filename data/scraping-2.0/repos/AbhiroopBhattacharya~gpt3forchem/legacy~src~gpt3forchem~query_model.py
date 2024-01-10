import openai
import time


def query_gpt3(model, df, temperature=0, max_tokens=10):
    completions = []

    for i, row in df.iterrows():
        completion = openai.Completion.create(
            model=model,  # "ada:ft-epfl-2022-06-18-13-55-33",
            prompt=row["prompt"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        completions.append(completion)
        time.sleep(5)

    return completions


def extract_prediction(completion):
    return completion["choices"][0]["text"].split("@")[0].strip()
