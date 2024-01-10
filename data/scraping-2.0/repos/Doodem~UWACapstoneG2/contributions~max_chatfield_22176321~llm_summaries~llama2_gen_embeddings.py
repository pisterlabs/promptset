from langchain.embeddings import LlamaCppEmbeddings

import pandas as pd

tenders_structured = pd.read_excel(r"../../../data/UpdatedAgainTenders.xlsx")
tenders_structured = tenders_structured[["Reference Number", "Contract Title", "Description"]].dropna(subset=["Reference Number"]).drop_duplicates()

from bs4 import BeautifulSoup
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text().replace('\xa0', ' ')
    cleaned = ' '.join(cleaned.split())
    return cleaned

import os
import pickle
import numpy as np
from tqdm import tqd

llama_save_path = "../../../../data/llama_embeddings/"
llama_embeddings = LlamaCppEmbeddings(model_path = "../../../../Models/llama-2-7b.Q4_K_M.gguf", n_ctx = 1024, n_batch = 32)

summary_path = "../../../data/extended_summaries/extended_summaries/"
with tqdm(total=len(tenders_structured.index), desc="Llama embeddings", colour='green') as pbar:
    for index, row in tenders_structured.iterrows():
        ref = str(row["Reference Number"])
        text_to_embed = row["Contract Title"] + ".  " + row["Description"]
        # check for existence of embedding
        emb_potential_path = os.path.join(llama_save_path, ref + ".npy")
        if not os.path.exists(emb_potential_path):
            # check if summaries are available
            summary_potential_path = os.path.join(summary_path, ref + ".pickle.sum")
            print(f"Generating {ref} embeddings...")
            summary_exists = os.path.exists(summary_potential_path)
            if summary_exists:
                text_to_embed = pickle.load(open(summary_potential_path, "rb"))["summary"]
                print("Summary available.")
            try:
                l_emb = llama_embeddings.embed_documents([text_to_embed])
                np.save(emb_potential_path, l_emb)
            except:
                print("Failed probably due to context length")
        pbar.update(1)
