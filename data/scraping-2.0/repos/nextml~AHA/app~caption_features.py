#
# Author: Scott Sievert (https://github.com/stsievert)
#
import os

import cytoolz as toolz
import numpy as np
import pandas as pd
import spacy
import torch
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from spacy.attrs import ORTH
from textblob import TextBlob

from . import syllabels_en

model = None
tokenizer = None
nlp = None
nlp_disable = ["parser", "tagger", "ner", "entityrecognizer", "textcat"]
SpacyDoc = spacy.tokens.doc.Doc


def load(small=False):
    """
    Load OpenAI model and NLP model

    Requires running

    > python -m spacy download en_core_web_lg
    """
    # Load pretrained model and tokenizer
    global model, tokenizer, nlp
    model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt").eval()
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    if small:
        nlp = spacy.load("en_core_web_sm")
    else:
        nlp = spacy.load("en_core_web_lg")
    return nlp


def perplexity(sentence):
    """
    Inputs
    ------
    sentenece : str
        Sentence to calculate the perplexity for.

    Returns
    -------
    perplexity : float
        The log of the probability (<= 0)

    Notes
    -----
    OpenAIGPTLHeadModel: causal/directional (takes sequence of words only
    preceding. BERT takes words before/after current word)

    It trains to maximize the log-probability:
       P(u_i | u_{i-k}, ..., u_{i-1})
    https://openai.com/blog/language-unsupervised/

    This model has 117M parameters. GPT2 has more parameters, but has some
    restrictions around release
    https://openai.com/blog/better-language-models/ (staged release section)

    """
    with torch.no_grad():
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss = model(tensor_input, lm_labels=tensor_input)
    return -1 * loss.item()


def get_joke_location(caption, full=False):
    """
    Parameters
    ----------
    caption : str

    Returns
    -------
    stats : dict
        Dictionary with describing stats. Includes key
        ``joke_location in range(4)`` which describes the quarter the joke is in.
    """
    blob = TextBlob(caption)
    ngrams = [ngram for ngram in blob.ngrams(n=4)]
    if len(ngrams) <= 2:
        ngrams = [ngram for ngram in blob.ngrams(n=2)]
    if len(ngrams) == 0:
        return {}
    perplexities = [perplexity(" ".join(ngram)) for ngram in ngrams]
    perplexities = np.array(perplexities)
    idx = np.argmin(perplexities)

    # Between 1 and 4
    phrase = " ".join(ngrams[idx])
    word = str(ngrams[idx][-1])
    joke_words = 1
    # idx += 3 # because 4-gram
    frac = idx / (len(ngrams))
    joke_quarter = int(frac * 4) + 1

    doc = nlp(caption, disable=["tagger", "ner", "entityrecognizer", "textcat"])
    noun_phrases = [str(x) for x in doc.noun_chunks]
    if any(word in bnp for bnp in noun_phrases):
        idx = [i for i, bnp in enumerate(noun_phrases) if word in bnp][0]
        joke_words = len(noun_phrases[idx].split(" "))
        phrase = noun_phrases[idx]

    kwargs = (
        {"word": "".join(word), "phrase": phrase, "noun_phrases": noun_phrases}
        if full
        else {}
    )
    return {
        "joke_quarter": joke_quarter,
        "joke_words": joke_words,
        "min_perplexity": perplexities.min().item(),
        "max_perplexity": perplexities.max().item(),
        "mean_perplexity": perplexities.mean().item(),
        "median_perplexity": np.median(perplexities).item(),
        **kwargs,
    }


def sim(w, S):
    """
    Inputs
    ------
    w : spacy Token
    S : spacy Doc

    Returns
    -------
    max_sim : float
        Maximal similarity between w and any token in S
    """
    if not w:
        return 0
    if not w.vector_norm:
        return 0
    sims = [s.similarity(w) for s in S]
    return max(sims)


def get_similarity(doc: SpacyDoc, C: SpacyDoc, A: SpacyDoc) -> dict:
    """
    Get stats on joke geometry.

    Inputs
    ------
    doc : SpacyDoc
        Caption
    C : SpacyDoc
        Context
    A : SpacyDoc
        Anomaly

    Returns
    -------
    info : Dict[str, float]
    """
    context_sims = [sim(j, C) for j in doc]
    anom_sims = [sim(j, A) for j in doc]
    diffs = [abs(sim(j, A) - sim(j, C)) for j in doc]
    diffs = np.array(diffs)
    if len(diffs) == 0:
        diffs = np.array([0])
    if len(context_sims) == 0:
        context_sims = np.array([0])
    if len(anom_sims) == 0:
        anom_sims = np.array([0])
    return {
        "sim_context_max": max(context_sims),
        "sim_anomaly_max": max(anom_sims),
        "sim_diff_max": diffs.max(),
        "sim_diff_mean": diffs.mean(),
        "sim_diff_median": np.median(diffs),
        "sim_diff_90_percentile": np.percentile(diffs, 90),
    }


def length_stats(text):
    doc = nlp(text, disable=nlp_disable)
    blob = TextBlob(text)
    counts = doc.count_by(ORTH)
    words = text.split(" ")
    return {
        "num_words": sum(counts.values()),
        "num_chars": sum(c.isdigit() or c.isalpha() for c in text),
        "num_sents": len(list(blob.sentences)),
        "num_syllables": sum(syllabels_en.count(word) for word in words),
    }


def readability(text):
    data = length_stats(text)
    sentences = data["num_sents"]
    words = data["num_words"]
    characters = data["num_chars"]
    syllabels = data["num_syllables"]

    def ARI(text):
        """
        1: kindegarten
        14: college
        """
        if (words == 0) or (sentences == 0):
            return 0
        return 4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43

    def flesch(text):
        """
        30-0: college level
        100-90: 5th grade
        """

        if (words == 0) or (sentences == 0):
            return 100

        return 206.835 - 1.015 * (words / sentences) - 84.6 * (syllabels / words)

    return {"readability_ARI": ARI(text), "readability_flesch": flesch(text), **data}


def spacy_stats(caption):
    doc = nlp(caption)
    tokens = [token for token in doc]
    POS = ["POS_" + token.pos_ for token in tokens]
    tags = ["TAG_" + token.tag_ for token in tokens]
    ents = ["ENT_" + ent.label_ for ent in doc.ents]

    is_blank = {
        k: sum(getattr(token, k) for token in tokens)
        for k in [
            "is_digit",
            "is_lower",
            "is_upper",
            "is_title",
            "is_punct",
            "is_currency",
            "like_num",
            "is_oov",
            "is_stop",
        ]
    }

    return {
        "num_stop": sum(t.is_stop for t in tokens),
        "num_alpha": sum(t.is_alpha for t in tokens),
        "num_tokens": len(tokens),
        "num_noun_chunks": len(list(doc.noun_chunks)),
        "num_words": len(doc),
        **toolz.frequencies(POS),
        **toolz.frequencies(tags),
        **toolz.frequencies(ents),
        **is_blank,
    }


def textblob_stats(caption):
    """ blob.sentiment trained on pos/neg movie reviews """
    blob = TextBlob(caption)
    proper_nouns = ["NNP" in tag for _, tag in blob.tags]
    lens = [len(noun_phrase.split(" ")) for noun_phrase in blob.noun_phrases]
    return {
        "num_proper_nouns": sum(proper_nouns),
        "num_noun_phrases": len(blob.noun_phrases),
        "max_len_noun_phrase": 0 if len(lens) == 0 else max(lens),
        "sentiment_subjectivity": blob.sentiment.subjectivity,
        "sentiment_polarity": blob.sentiment.polarity,
        "num_sentences": len(blob.sentences),
    }


def stats(caption, C, A):
    """
    Parameters
    ----------
    caption : str
    C : SpacyDoc
        Describes the context.
    A : SpacyDoc
        Describes the anomaly.

    Returns
    -------
    features : dict
        Features for this caption+context+anomly.

    Notes
    -----
    ``C`` and ``A`` can be  generated with

        nlp(" ".split(context_words))

    """
    caption_doc = nlp(caption, disable=nlp_disable)
    d1 = textblob_stats(caption)
    d2 = spacy_stats(caption)
    d3 = readability(caption)
    d4 = get_joke_location(caption)
    d5 = get_similarity(caption_doc, C, A)
    return {**d1, **d2, **d3, **d4, **d5}


def collect_df_stats(df, context, anom):
    contests = df.contest.unique()
    print(contests)
    c = contests[0]
    fname = f"{c}-nlp2-features.csv"
    DIR = "nlp2-features/"
    if fname in os.listdir(DIR):
        return False
    load()
    C = nlp(" ".join(context), disable=nlp_disable)
    A = nlp(" ".join(anom), disable=nlp_disable)
    s = df.caption.apply(stats, args=(C, A))
    t = pd.DataFrame(dict(s)).T
    assert t.shape[0] == len(df)
    out = pd.concat((df, t), axis="columns", sort=False)
    assert out.shape[0] == len(df)
    out.drop(
        columns=["rank", "funny", "unfunny", "somewhat_funny", "count"], inplace=True
    )
    out.to_csv(DIR + fname, index=False)
    return True


if __name__ == "__main__":
    caption = "This is a really big iPhone from Apple!"
    s = stats(caption)
    from pprint import pprint

    pprint(s)
    print(len(s))
