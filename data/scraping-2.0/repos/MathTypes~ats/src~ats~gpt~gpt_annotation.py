import argparse
import logging
import os
import pandas as pd

from ratelimiter import RateLimiter
from promptify import OpenAI
from promptify import Prompter

from neo4j_util.sentiment_api import get_gpt_unprocessed_replied_tweets
from neo4j_util.neo4j_tweet_util import Neo4j
from util import logging_utils
import openai


@RateLimiter(max_calls=25, period=60)
def query(prompt, data, to_print=True):
    example = """
    Desired format: semicolon separated list. each list is colon separated list of entity name, entity class, sentiment and score.
    For example: "WFC,EQUITY,POSITIVE,5;BAC,EQUITY,NEGATIVE,1,mid Jan,DATE,POSITIVE,2"
    """
    prompt = "{}\n{}\ntext:{}".format(prompt, example, data)
    logging.info(f"prompt:{prompt}")
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    payload = response["choices"][0]["text"].strip()
    if to_print:
        print(payload)
    return payload


def get_named_entity(x: str):
    res = query("Extract top three financial named entities from following text", x)
    return res


def get_equity_sentiment(x: str):
    res = query(
        "Extract the important financial entities mentioned in the text below along with sentiment. First extract all company names, then extract all people names, then extract specific topics which fit the content and finally extract general overarching themes",
        x,
    )
    return res


def parse_csv(data, names):
    lines = data.split(";")
    names_dict = {name: [] for name in names}
    for line in lines:
        if line:
            cols = line.split(",")
        for i, col in enumerate(cols):
            names_dict[names[i]].append(col)
    logging.info(f"dict:{names_dict}")
    df = pd.DataFrame(names_dict)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to add GPT annotation")
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    api_key = os.environ["OPENAI_API_KEY"]
    model = OpenAI(api_key)  # or `HubModel()` for Huggingface-based inference
    nlp_prompter = Prompter(model)

    logging_utils.init_logging()
    while True:
        data = get_gpt_unprocessed_replied_tweets()
        logging.error(f"unprocess_data:{data}")
        neo4j_util = Neo4j()
        for i, row in data.iterrows():
            logging.info(f'link:{row["perma_link"]}')
            logging.info(f'text:{row["text"]}')
            # result = get_named_entity(row["text"])
            # logging.info(f'result:{result}')
            # entity_df = parse_csv(result, ["entity_name"])
            # logging.info(f'entity_df:{entity_df}')
            result = get_equity_sentiment(row["text"])
            logging.info(f"result:{result}")
            sentiment_df = parse_csv(
                result,
                ["entity_name", "entity_class", "sentiment_class", "sentiment_score"],
            )
            sentiment_df["tweet_id"] = row["tweet_id"]
            logging.info(f"sentiment_df:{sentiment_df}")
            # Output
            # logging.info(f'equity_sentiment_result:{equity_sentiment_result}')
            # logging.error(f'process_data:{data}')
            neo4j_util.update_gpt_entities(sentiment_df)
