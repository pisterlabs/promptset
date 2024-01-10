import argparse

import cohere
import numpy as np

import evaluation


co = cohere.Client("gr3eALiN829VBG4rWgT7YrCFZTNZlEoHDIYLdpaP")
embed_dim = {"small": 1024, "large": 4096, "multilingual-22-12": 768}


def text2vec(texts, model="multilingual-22-12"):
    assert model in embed_dim.keys()

    embeds = co.embed(texts=texts, model=model)
    embeds = np.asarray(embeds.embeddings, dtype=np.float32)
    return embeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("texts", nargs="+", type=str)
    args = parser.parse_args()

    texts = args.texts
    embeds = text2vec(texts=texts)
    for i in range(embeds.shape[0]):
        for j in range(i + 1, embeds.shape[0]):
            embed_1 = embeds[i]
            embed_2 = embeds[j]
            score = evaluation.cos_sim(embed_1, embed_2)
            print(f"{texts[i]}\t{texts[j]}\t{score}")
