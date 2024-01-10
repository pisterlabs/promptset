"""
Classifies the different datasets used using GPT-3.5 and a huggingface transformer.
"""
import re
import os

import pandas as pd
import numpy as np
import torch
import openai
from tenacity import retry, stop_after_attempt, wait_fixed
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from _pipelines import topic_pipelines, sentiment_pipelines

device = torch.device("cpu")
openai.api_key = os.environ["OPENAI_API_KEY"]

SYSTEM_PROMPT = """You are a sophisticated AI model trained to classify news headlines \
for a financial hedge fund.

Your main task is to categorize each headline based on two criteria:
1. The primary subject matter: Is it predominantly about Finance/Economics or \
another unrelated subject? Respond with 'Economics' or 'Other'.
    - Economic headlines generally cover topics such as financial markets, business, \
financial assets, trade, employment, GDP, inflation, or fiscal and monetary policy.
    - Non-economic headlines might include sports, entertainment, politics, science, \
weather, health, or other unrelated news events.
2. The overall sentiment conveyed: Does the headline convey a Positive, Neutral, or \
Negative sentiment with regard to the current state or potential future impact on \
the economy or the asset described?
    - Positive sentiment headlines suggest growth, improvement, or stability in \
economic conditions.
    - Neutral sentiment headlines do not clearly indicate a positive or negative \
impact on the economy.
    - Negative sentiment headlines imply economic decline, uncertainty, or \
unfavorable conditions.

Your response should follow this format: 'Number. Topic, Sentiment'.

Here are some examples to guide you:
1. For 'Economic growth expected to surge this year', respond with '1. Economics, Positive'
2. For 'Wenger attacks Madrid for transfer rule-bending', respond with '2. Other, Negative'
3. For 'Is the improved economy really making us happier?', respond with '3. Economics, Neutral'
4. For 'Thomas Edison Voted Most Iconic Inventor In U.S. History', respond with '4. Other, Positive'
5. For 'Major tech company under federal investigation', respond with '5. Economics, Negative'
6. For 'The transaction doubles Tecnomens workforse, and adds a fourth to their net sales.', \
respond with '6. Economics, Positive'
"""


def process_data_pipeline(pipeline, file_path, errors='raise'):
    """Process data pipeline function

    This function takes a data processing pipeline and a file path as input,
    applies the pipeline to the file and saves the resulting data to a CSV file.
    It returns the total number of tokens in the processed data..

    Args:
        pipeline (function): A function that takes a file path and returns a pandas DataFrame.
        file_path (str): The path to the input file.
        cols (list): A list of column names to include in the output file.

    Returns:
        int: The total number of tokens processed.
    """
    file_name = os.path.basename(file_path)
    file_name = re.sub(r"\.\w+$", "", file_name)
    file_name += ".csv"
    df = pipeline(file_path)
    df = df.reset_index(drop=True)
    cols = df.columns.tolist()
    cols.extend(["GPT Topic", "CardiffNLP Topic", "Topic_04 Topic",
                 "GPT Sentiment", "FinRoberta Sentiment"])

    if not os.path.exists('./data/classified-data'):
        os.makedirs('./data/classified-data')
    output_file_path = os.path.join(r"./data/classified-data", file_name)
    _touch_file(output_file_path, cols=cols, errors=errors)

    chunk_size = 20
    total_tokens = 0
    for chunk, tokens in _classify_chunks(df, chunk_size):
        try:
            chunk.to_csv(output_file_path, mode='a', header=False, index=False)
        except Exception as e:
            print(e)
        total_tokens += tokens
    return total_tokens


def _touch_file(file_path, cols, errors='raise'):
    """Create a new file with the given columns"""
    if os.path.exists(file_path):
        if errors == 'ignore':
            return
        else:
            raise FileExistsError(f"File {file_path} already exists.")
    else:
        str_cols = ",".join(cols)
        with open(file_path, 'w', newline='') as f:
            f.write(f"{str_cols}\n")


def _classify_chunks(df, chunk_size):
    chunks = _split_df(df, chunk_size)
    sentiment_transformer = _launch_sentiment_transformer()
    topic_transformers = _launch_topic_transformers()
    for chunk in chunks:
        yield _process_chunk(chunk, sentiment_transformer, topic_transformers)


def _split_df(df, chunk_size):
    return np.array_split(df, len(df) // chunk_size + 1)


def _launch_sentiment_transformer():
    model = AutoModelForSequenceClassification.from_pretrained(
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    )
    return model, tokenizer


def _launch_topic_transformers():
    cardiff_model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/tweet-topic-21-multi").to(device)
    cardiff_model.eval()
    cardiff_tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/tweet-topic-21-multi")

    topic_04_model = AutoModelForSequenceClassification.from_pretrained(
        "jonaskoenig/topic_classification_04", from_tf=True).to(device)
    topic_04_model.eval()
    topic_04_tokenizer = AutoTokenizer.from_pretrained(
        "jonaskoenig/topic_classification_04")
    return (cardiff_model, cardiff_tokenizer), (topic_04_model, topic_04_tokenizer)


def _process_chunk(chunk, sentiment_pipeline, topic_pipeline):
    chunk = chunk.reset_index(drop=True)
    headlines = chunk['Headlines'].tolist()

    gpt_output, tokens = _classify_headlines_gpt(headlines)
    gpt_df = _parse_gpt_output(gpt_output)
    transformer_df = _classify_headlines_transformers(headlines, sentiment_pipeline, topic_pipeline)

    try:
        df = pd.DataFrame({
            'GPT Topic': gpt_df['GPT Topic'],
            'CardiffNLP Topic': transformer_df['CardiffNLP Topic'],
            'Topic_04 Topic': transformer_df['Topic_04 Topic'],
            'GPT Sentiment': gpt_df['GPT Sentiment'],
            'FinRoberta Sentiment': transformer_df['FinRoberta Sentiment'],
        })
        df = pd.concat([chunk, df], axis=1)
    except Exception as e:
        print(repr(e))
        df = None
    return df, tokens


def _classify_headlines_transformers(headlines, sentiment_pipeline, topic_pipelines):
    """Classify headlines using existing Huggingface Transformers

    This function uses HuggingFace models to classify the sentiment and topic of news
    headlines. The function tokenizes the input headlines using a provided tokenizer,
    pads and truncates the sequences, and passes them through the Transformer model to
    generate sentiment predictions.
    """
    cardiff_pipeline, topic_04_pipeline = topic_pipelines
    CARDIFF_LABEL = 'business_&_entrepreneurs'
    TOPIC_04_LABEL = 'Business & Finance'

    cardiff_topic = __classify_topic_transformer(headlines, *cardiff_pipeline, CARDIFF_LABEL)
    topic_04_topic = __classify_topic_transformer(headlines, *topic_04_pipeline, TOPIC_04_LABEL)
    finroberta_sentiment = __classify_sentiment_transformer(headlines, *sentiment_pipeline)

    df = pd.DataFrame({
        'CardiffNLP Topic': cardiff_topic,
        'Topic_04 Topic': topic_04_topic,
        'FinRoberta Sentiment': finroberta_sentiment,
    })
    return df


def __classify_sentiment_transformer(headlines, model, tokenizer):
    headlines = tokenizer(headlines, add_special_tokens=True, max_length=30,
                          padding="max_length", truncation=True, return_tensors="pt")
    headlines = headlines.to(device=device)
    output = model(**headlines)
    probs = torch.nn.functional.softmax(output.logits, dim=1)
    preds = probs.argmax(dim=1)
    preds = pd.Series(preds.to('cpu').numpy()).map(model.config.id2label)
    preds = preds.str.capitalize()
    return preds


def __classify_topic_transformer(headlines, model, tokenizer, econ_label):
    headlines = tokenizer(headlines, add_special_tokens=True, max_length=30,
                          padding="max_length", truncation=True, return_tensors="pt")
    headlines = headlines.to(device=device)
    output = model(**headlines)
    probs = torch.sigmoid(output.logits)
    ECON_INDEX = model.config.label2id[econ_label]
    preds = (probs[:, ECON_INDEX] > 0.8)

    MAPPING = {True: 'Economics', False: 'Other'}
    preds = pd.Series(preds.to('cpu').numpy()).map(MAPPING)
    return preds


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def _classify_headlines_gpt(headlines):
    """Classify headlines using GPT-3.5 function

    This function uses the GPT-3.5 model to classify the topic and sentiment of news
    headlines. The function constructs a user prompt that includes the headlines to
    be classified and sends it to the GPT-3.5 model for completion. The completed
    text response is parsed to extract the topic and sentiment labels.

    Parameters
    ----------
    headlines : list
        A list of strings containing the news headlines to be classified.

    Returns
    -------
    output : str
        A string containing the completed response from the GPT-3.5 model.
    total_tokens : int
        An integer representing the number of tokens used to generate the response.
"""
    user_prompt = "Classify the following headlines:"
    for idx, headline in enumerate(headlines):
        if not headline.startswith('"'):
            headline = f'"{headline}"'
        user_prompt += f'\n{idx + 1}. {headline}'

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        max_tokens=300,
        temperature=0.2,
        frequency_penalty=-0.1,
        presence_penalty=-0.1
    )
    return response['choices'][0]['message']['content'], response['usage']['total_tokens']


def _parse_gpt_output(text):
    """Parse GPT output

    This function takes a string of text containing GPT output as input,
    and extracts the topics and sentiment labels from the text. It then processes
    the labels to ensure they are in the correct format.

    Parameters
    ----------
    text : str
        A string of text containing GPT output.

    Returns
    --------
    topics : list
        A list of strings representing the topics of the headlines in the GPT output.
    sentiment : list
        A list of strings representing the sentiment labels of the headlines in the GPT output.
    """
    headlines = text.split("\n")
    try:
        topics = [h.split(",")[0].split(".")[1].strip() for h in headlines]
        sentiment = [h.split(",")[1].strip() for h in headlines]
    except IndexError:
        print(f'IndexError: {text}')
        topics = ["Neither"] * len(headlines)
        sentiment = ["Neutral"] * len(headlines)

    for idx, c in enumerate(topics):
        if 'Economy' in c or 'Compan' in c or 'Finance' in c or 'Real Estate' in c:
            topics[idx] = 'Economics'
        elif c == 'Neither':
            topics[idx] = 'Other'
        elif c not in {'Economics', 'Other'}:
            print(f'Topic: {c}')
            topics[idx] = 'Other'

    for idx, s in enumerate(sentiment):
        if s == "Bullish" or 'Positive' in s:
            sentiment[idx] = "Positive"
        elif s == "Bearish" or 'Negative' in s:
            sentiment[idx] = "Negative"
        elif s in ("Neither", "Mixed", "Other") or 'Neutral' in s:
            sentiment[idx] = "Neutral"
        elif s not in ["Positive", "Negative", "Neutral"]:
            print(f'Sentiment: {s}')
            sentiment[idx] = "Neutral"

    df = pd.DataFrame({'GPT Topic': topics, 'GPT Sentiment': sentiment})
    return df


if __name__ == '__main__':
    # WARNING: This script will take a very long time to run (~1 day on my system)
    total_tokens = 0
    pipelines = {**topic_pipelines, **sentiment_pipelines}
    for name, pipeline in pipelines.items():
        total_tokens += process_data_pipeline(
            pipeline['pipeline'],
            pipeline['file_path'],
            errors='raise'
        )
        print(f"{name} done!")
    print(f"Total tokens: {total_tokens}")
    # $0.002 per 1000 token as of 2023-05-09
    print(f'Price: $ {total_tokens / 1000 * 0.002:.2f}')
