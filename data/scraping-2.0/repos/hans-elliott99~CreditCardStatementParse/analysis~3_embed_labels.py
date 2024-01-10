import os
import openai
import pickle
from pathlib import Path
from datetime import datetime
import json

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves",
              "you", "your", "yours", "yourself", "yourselves", "he", "him",
              "his", "himself", "she", "her", "hers", "herself", "it", "its",
              "itself", "they", "them", "their", "theirs", "themselves", "what",
              "which", "who", "whom", "this", "that", "these", "those", "am",
              "is", "are", "was", "were", "be", "been", "being", "have", "has",
              "had", "having", "do", "does", "did", "doing", "a", "an", "the",
              "and", "but", "if", "or", "because", "as", "until", "while", "of",
              "at", "by", "for", "with", "about", "against", "between", "into",
              "through", "during", "before", "after", "above", "below", "to",
              "from", "up", "down", "in", "out", "on", "off", "over", "under",
              "again", "further", "then", "once"]


# TODO:
# using just the class name, or the class name + my own made up description, does
# much better than class name + long list of keywords (which completely fails)
# - figure out something in between
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

def load_labels(fp):
    with open(fp, "r") as f:
        l = json.load(f)
    return l

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

def _concat_kws(labels, concat_kws):
    return [k + ": " + concat_kws.join(v) for k, v in labels.items()]

def embed_labels(labels, save_fp, include_keywords=True, concat_keywords=", "):
    """Embed the class labels.
    labels - a dict of {class-name: optional description}s
    concat_descr - if True, include the description in the prompt sent to openai
    save_fp - if provided, save pickled embeddings here

    Returns a dict of {class-name: embeddings}
    """
    if include_keywords:
        labs = _concat_kws(labels, concat_kws=concat_keywords)
    else:
        labs = list(labels.keys())
    # get embeddings from openai
    lab_embs = {k: get_embedding(lab) for k, lab in zip(labels.keys(), labs)}
    # pickle embeddings
    with open(save_fp, "wb") as f:
        pickle.dump(lab_embs, f)
    return lab_embs


if __name__=="__main__":

    # cli args
    labels_fp = "analysis/data/labels/labels_nl_descr_simple.json"
    out_dir = "analysis/data"
    suffix = "date"
    concat_keywords = ", "
    uniq_keywords = False


    # set openai key
    apikey = os.environ.get("OPENAI")
    if apikey is None:
        raise RuntimeError("Required environmental variable 'OPENAI' is not set.")
    openai.api_key = os.environ["OPENAI"]

    # args
    base = str(Path(labels_fp).name).replace(".json", "")
    if suffix == "date":
        suffix = datetime.today().strftime('%Y-%m-%d')
    if suffix:
        suffix = "_" + suffix
    out_kw = Path(out_dir) / f"{base}{suffix}.pkl"
    out_smpl = Path(out_dir) / f"labs_only{suffix}.pkl"

    # load labels
    labels = load_labels(labels_fp)
    if uniq_keywords:
        # just keep unique words from each keyword list
        labels = {k : set(" ".join(v).split(" ")) for k, v in labels.items()}
        labels = {k : sorted([v for v in vals if v not in stop_words]) 
                      for k, vals in labels.items()}
    
    # print info
    labs_w_kws = _concat_kws(labels, concat_keywords)
    print("\n***TEXT TO EMBED")
    print("***")
    for i, l in enumerate(labels.keys()):
        print(f"***LABEL: {l}")
        print("***WITH KEYWORDS:")
        print(f"   {labs_w_kws[i]}")
        print("***")

    print_tokens(text=" ".join(labs_w_kws))


    # EMBED LABELS
    if ask_yesno("Embed labels with keywords?"):
        # inlude keywords
        print("***Embedding...")
        embed_labels(labels, save_fp=out_kw,
                     include_keywords=True, concat_keywords=concat_keywords)
        print(f"***Saved {out_kw}")
        print("***")

    if ask_yesno("Embed labels without keywords?"):
        # class names only
        print("***Embedding...")
        embed_labels(labels, save_fp=out_smpl, include_keywords=False)
        print(f"Saved {out_smpl}")