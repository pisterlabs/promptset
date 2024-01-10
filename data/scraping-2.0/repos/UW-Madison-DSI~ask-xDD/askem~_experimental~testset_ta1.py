import ast
import os
from typing import List, Optional

import gspread
import openai
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


def load_testset() -> pd.DataFrame:
    """Load testset from Google Sheet."""

    GCP_SECRET_FILE_PATH = os.getenv("GCP_SECRET_FILE_PATH")

    gc = gspread.service_account(filename=GCP_SECRET_FILE_PATH)
    sheet = gc.open("ASKEM-TA1-testset").worksheet("questions")

    records = sheet.get_values()
    labels = records[0]
    data = records[1:]

    new_labels = [label.lower().replace(" ", "_") for label in labels]
    df = pd.DataFrame.from_records(data, columns=new_labels)
    df["is_keyword"] = df["is_keyword"].astype(int)
    df["is_complex"] = df["is_complex"].map({"": 0, "1": 1}).astype(int)
    return df[["source", "target_type", "is_keyword", "is_complex", "question"]]


def gpt_eval(result: str, model: str = "gpt-4") -> List[str]:
    """Eval the eval result with GPT-4."""

    system_message = {
        "role": "system",
        "content": "You are a expert in epidemiology. Given the following evaluation results, select the best API for the given question. Tie is allowed. You organize your output like this: ['API1', 'API2', 'API3'] returning one or more best APIs. Return 'None' if you think none of the APIs are good.",
    }

    user_message = {
        "role": "user",
        "content": f"Given this results: {result}, which API is the best?",
    }

    response = openai.ChatCompletion.create(
        model=model, messages=[system_message, user_message], temperature=0.9
    )

    return ast.literal_eval(response["choices"][0]["message"]["content"])
