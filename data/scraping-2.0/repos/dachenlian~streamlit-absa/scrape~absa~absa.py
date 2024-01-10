import asyncio
from collections import Counter
from pathlib import Path
from string import Template
import re

from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties
import openai
import pandas as pd
import seaborn as sns
import tiktoken
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm

from scrape.absa.types import CountABSAOutput, GetABSAOutput, GetAnnotatedABSAOutput
from scrape.absa.prompts import ANNOTATED_ABSA_PROMPT
from scrape.types import OpenAIModel
from scrape.utils import chunker


BASE = Path(__file__).resolve().parent.parent.parent
font_path = str(BASE / "data/fonts/SourceHanSerifK-Light.otf")
fontManager.addfont(font_path)
prop = FontProperties(fname=font_path)
sns.set(font=prop.get_name())

enc = tiktoken.get_encoding("cl100k_base")


def get_absa(
    client: openai.Client,
    texts: list[str],
    base_prompt: Template,
    main_body: str = "",
    max_length: int = 2000,
    model_name: OpenAIModel = OpenAIModel.GPT3_5,
    chunk_size: int = 25,
) -> GetABSAOutput:
    """
    Retrieves Aspect-Based Sentiment Analysis (ABSA) for a list of texts using the OpenAI GPT model.

    Args:
        client (openai.Client): The OpenAI client used to make API calls.
        texts (list[str]): The list of texts to perform ABSA on.
        base_prompt (str): The base prompt used for each text.
        main_body (str | None, optional): The main body of the prompt. Defaults to None.
        model_name (OpenAIModel, optional): The name of the OpenAI model to use. Defaults to OpenAIModel.GPT3_5.
        chunk_size (int, optional): The number of texts to process in each API call. Defaults to 25.

    Returns:
        GetABSAOutput: The ABSA output containing the sentiment analysis results for each text.
    """
    from scrape.utils import call_model

    _texts = [f"Text {i}:\n{t}" for i, t in enumerate(texts, start=1)]
    texts = []
    length = 0
    for t in _texts:
        texts.append(t)
        length += len(enc.encode(t))
        if length > max_length:
            break

    responses = {}
    chunked = chunker(texts, chunk_size)
    logger.info(f"Chunked into {len(chunked)} chunks")
    tasks = []
    for c in tqdm(chunked):
        prompt = base_prompt.substitute(text="\n".join(c))
        if main_body:
            prompt = "main_body: " + main_body + "\n\n" + prompt

        messages = [{"role": "user", "content": prompt}]

        response = call_model(client, messages, model_name, return_json=True)
        responses = responses | response
        # tasks.append(call_model(client, messages, model_name, return_json=True))
    
    # results = wait tqdm_asyncio.gather(*tasks)

    # for result in results:
    #     responses = responses | result

    responses = sorted(
        responses.items(), key=lambda x: int(re.split(r"[_\s]", x[0])[1])
    )
    return responses


def get_annotated_absa(
    client: openai.Client,
    text: str,
    aspects: list[str],
    base_prompt: Template = ANNOTATED_ABSA_PROMPT,
    max_length: int = 2000,
    model_name: OpenAIModel = OpenAIModel.GPT3_5,
) -> GetAnnotatedABSAOutput:
    """
    Retrieves annotated aspect-based sentiment analysis (ABSA) for the given text and aspects.

    Args:
        client (openai.Client): The OpenAI client used to make API calls.
        text (str): The input text for ABSA.
        aspects (list[str]): The list of aspects to analyze in the text.
        base_prompt (str): The base prompt for the ABSA model.
        model_name (OpenAIModel, optional): The name of the OpenAI model to use for ABSA. Defaults to OpenAIModel.GPT3_5.

    Returns:
        GetAnnotatedABSAOutput: The annotated ABSA output.

    """
    from scrape.utils import call_model

    text = enc.decode(enc.encode(text)[:max_length])

    prompt = base_prompt.substitute(text=text, aspects=", ".join(aspects))
    messages = [{"role": "user", "content": prompt}]
    _response = call_model(client, messages, model_name=model_name, return_json=True)
    _response = _response["text"]
    response = []
    for r in _response:
        if isinstance(r, list):
            r = tuple(r)
        response.append(r)
    return response


def get_val_from_absa_output_key(
    output: GetABSAOutput, key: str
) -> dict[int, bool | str]:
    """
    Retrieves the values associated with a given key from the ABSA output.

    Args:
        output (GetABSAOutput): The ABSA output.
        key (str): The key to retrieve the values for.

    Returns:
        dict[int, bool | str]: A dictionary mapping the index to the value associated with the key.
    """
    d = {}
    for r in output:
        idx = int(re.split(r"[_\s]", r[0])[1]) - 1
        try:
            d[idx] = r[1][key]
        except KeyError:
            d[idx] = None
    return d


def count_absa(output: GetABSAOutput) -> CountABSAOutput:
    """
    Counts the number of positive, negative, and neutral sentiments for each aspect in the given output.

    Args:
        output (GetABSAOutput): The output of the get_absa function.

    Returns:
        CountABSAOutput: An object containing the counts of positive, negative, and neutral sentiments for each aspect.
    """
    positive = Counter()
    negative = Counter()
    neutral = Counter()

    for _, review in output:
        for aspect, sentiment in review.items():
            aspect = aspect.lower().replace("_", " ")
            if sentiment == "positive":
                positive[aspect] += 1
            elif sentiment == "negative":
                negative[aspect] += 1
            elif sentiment == "neutral":
                neutral[aspect] += 1

    return CountABSAOutput(positive=positive, negative=negative, neutral=neutral)


def create_absa_counts_df(
    counts: CountABSAOutput, proportional: bool = True
) -> pd.DataFrame:
    """
    Create a DataFrame from the counts of positive, negative, and neutral sentiments.

    Args:
        counts (CountABSAOutput): The counts of positive, negative, and neutral sentiments.
        proportional (bool, optional): Whether to calculate the proportions of each sentiment. Defaults to True.

    Returns:
        pd.DataFrame: The DataFrame containing the counts or proportions of each sentiment.
    """
    positive = pd.Series(counts.positive, dtype=int)
    negative = pd.Series(counts.negative, dtype=int)
    neutral = pd.Series(counts.neutral, dtype=int)

    df = pd.DataFrame(
        {"positive": positive, "negative": negative, "neutral": neutral}
    ).fillna(0)

    total = df.sum(axis=1)

    if proportional:
        df = df.div(df.sum(axis=1), axis=0)

    df["total"] = total

    return df


def create_absa_df(data: GetABSAOutput) -> pd.DataFrame:
    """
    Create a pandas DataFrame from the GetABSAOutput data.

    Args:
        data (GetABSAOutput): The GetABSAOutput data containing user and sentiment information.

    Returns:
        pd.DataFrame: A DataFrame with user index, positive aspects, negative aspects, and neutral aspects.
    """
    res = []
    for user, d in data:
        idx = int(re.split(r"[_\s]", user)[-1]) - 1
        pos, neg, neu = [], [], []
        for aspect, sentiment in d.items():
            aspect = aspect.lower().replace("_", " ")
            if sentiment == "positive":
                pos.append(aspect)
            elif sentiment == "negative":
                neg.append(aspect)
            elif sentiment == "neutral":
                neu.append(aspect)
        res.append(
            {
                "user_idx": idx,
                "positive": ", ".join(pos),
                "negative": ", ".join(neg),
                "neutral": ", ".join(neu),
            }
        )
    return pd.DataFrame(res).set_index("user_idx")


def create_absa_heatmap(
    df: pd.DataFrame, min_occurrences: int = 2, cmap: str = "rocket"
) -> Figure:
    """
    Create a heatmap of the given DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to create the heatmap from.
        cmap (str): The colormap to use for the heatmap. Default is "rocket".

    Returns:
        Figure: The generated heatmap figure.
    """
    df[df["total"] >= min_occurrences]
    aspects = pd.unique(df[["positive", "negative", "neutral"]].values.ravel("K"))
    height = len(aspects) * 1.1
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(df, cmap=cmap, annot=True, fmt=".2f")
    return fig
