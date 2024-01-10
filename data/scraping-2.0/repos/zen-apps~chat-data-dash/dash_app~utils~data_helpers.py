"""
Data functions to help with data processing
"""

import pandas as pd
import base64
import io
from langchain.schema import ChatMessage


def sample_dataframe(df):
    num_rows = df.shape[0]
    if num_rows < 10:
        sample_size = num_rows  # if less than 10 rows, use all of them
    elif num_rows < 1000:
        sample_size = 10
    elif num_rows < 10000:
        sample_size = 20
    else:
        sample_size = 30

    return df.sample(n=sample_size, random_state=5, replace=False, frac=None)


def upload_file_processing(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        sampled_df = sample_dataframe(df)
        table_name = filename.split(".")[0]
        return df, table_name, sampled_df


def dict_to_chat_obj(dict_chat):
    if dict_chat["user"]:
        return ChatMessage(role="user", content=dict_chat["message"])
    else:
        return ChatMessage(role="assistant", content=dict_chat["message"])
