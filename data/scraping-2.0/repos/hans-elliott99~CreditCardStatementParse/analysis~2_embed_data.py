import os
import openai
import pandas as pd
from pathlib import Path


def ask_yesno(msg):
    msg2 = f"{msg} (yes/no): "
    i = input(msg2).strip().lower()
    if i.startswith("y"):
        return True
    elif i.startswith("n"):
        return False
    else:
        print("ERROR: Answer y(es) or n(o) to continue execution.")
        ask_yesno(msg)

def calc_tokens(text, token_rate=0.0001):
    tok = len(''.join(text)) / 4
    return tok, tok * token_rate

def print_tokens(text):
    token_rate = 0.0001
    tok, cost = calc_tokens(text)
    print(f"***Embedding roughly {tok} tokens at rate of ${token_rate} per token...")
    print(f"*** ~ ${tok * token_rate}")


def get_embedding(text, engine="text-similarity-davinci-001", **kwargs):
    """Embed some text using an openai model.
    Requires that `openai.api_key = "your-api-key"` has been executed.
    Returns the embeddings as a list.
    """
    # https://platform.openai.com/docs/guides/embeddings/use-cases
    # https://platform.openai.com/docs/guides/embeddings/what-are-embeddings
    # https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py
    # new lines can negatively affect performance
    text = text.replace("\n", " ")
    emb = openai.Embedding.create(
        input=[text], engine=engine, **kwargs
    )
    return emb["data"][0]["embedding"]


def embed_df_column(df, txt_col):
    """Embed each row of a (string) column in a pd.DataFrame.
    Returns the updated df (although the provided df will be modified in-place)
    """
    df[txt_col + "_emb"] = df[txt_col].apply(lambda x: get_embedding(text=x))
    return df

if __name__=="__main__":


    # cli args
    fp = "./analysis/data/labeled_spend_01-08_2023.txt"
    delim = "|"
    txt_col = "descr"
    out_dir = "./analysis/data"

    # set openai key
    apikey = os.environ.get("OPENAI")
    if apikey is None:
        raise RuntimeError("Required environmental variable 'OPENAI' is not set.")
    openai.api_key = os.environ["OPENAI"]

    # Load data
    dat = pd.read_csv(fp, sep=delim, dtype=str)
    dat["txt_processed"] = dat[txt_col].str.strip().str.lower()

    # print info
    print("***TEXT TO EMBED (first 5 rows)")
    print(dat["txt_processed"].head(5))
    print("***")
    print_tokens(text=''.join(dat[txt_col]))
    print("***")

    # embed
    if ask_yesno(f"Embed text from column: '{txt_col}' ?"):
        print("***Embedding...")
        dat = embed_df_column(dat, txt_col)
        # save
        base = Path(fp).name.split(".")[0]
        out = Path(out_dir) / f"{base}_w_emb.csv"
        dat.to_csv(
            out, sep=delim, index=False 
        )
        print(f"***Saved to {out}")
