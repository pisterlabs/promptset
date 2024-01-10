from openai_utils import OpenAIAPI
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":

    openai = OpenAIAPI()
    openai.login()

    df = pd.read_csv("data/cathay_web_faq.csv")
    df.columns = ["prompt", "completion"]
    df = df[df["completion"].notnull()].reset_index(drop=True)
    df["prompt"] = df["prompt"].apply(lambda x: x.strip())
    df["completion"] = df["completion"].apply(
        lambda x: "".join([item.strip() for item in x.split("\n") if item.strip()])
    )

    wait = 5
    ada_embeddings = []
    for text in tqdm(df.completion.values):
        ebd = openai.get_embeddings(text, model="text-embedding-ada-002")
        ada_embeddings.append(ebd)

    df["ada_embeddings"] = ada_embeddings
    df.to_pickle("data/cathay_faq_embeddings.pkl")
