"""Select bee questions and answers using chatgpt in a RAG way

# Usage Example

gpt_model="gpt-3.5-turbo-16k"

prompt_template="My bee questions are stored inside a sqlite3 table: qa. It has the following columns and short descriptions: (1) 'qustion_no' is the question number; (2) 'question' is the question; (3) 'answer' is the answer. When I make a request below, I want you to write the sql query and also run the sql and get me the final output."

prompt="${prompt_template}. Now can you select 3 random questions and anssers?"

verbose=3

python rnd/ai/iac/serving/bee_qa.py --gpt_model=$gpt_model --prompt="$prompt" --verbose=$verbose

"""
from cafpyutils.generic import create_logger
import json
from openai import OpenAI
import os
import pandas as pd
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import List

from rnd.ai.calendar.utils.chat_utils import chat_with_backoff
from rnd.ai.iac.serving.save_pdfs import DbOperator

logger = create_logger(__name__)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", None)  # No truncation


def parse_sql_query(text: str) -> str:
    text = text.lower()

    # Grab the first ```sql and the immediate next ';'
    pattern = r"```.*?```"
    match = re.search(pattern, text, re.DOTALL)

    sql = text[match.start() : match.end()].replace("```", "").replace("sql", "")
    assert ("select" in sql) and ("from" in sql)
    return sql


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def chat_with_backoff(client, model: str, messages: List[dict]):
    """Backoff to combat with rate limits"""
    response = client.chat.completions.create(model=model, messages=messages)
    return response


def main(
    gpt_model: str,
    prompt: str,
    openai_api_key: str = os.environ.get("OPENAI_API_KEY"),
    verbose: int = 1,
) -> dict:
    openai_client = OpenAI(api_key=openai_api_key)

    response = chat_with_backoff(client=openai_client, model=gpt_model, messages=[{"role": "user", "content": prompt}])
    reply = response.choices[0].message.content
    logger.info("ChatGPT's reply: %s" % reply)

    logger.info("Parsing ChatGPT's sql query suggestion and getting the final result")
    sql_query = parse_sql_query(reply)
    op = DbOperator(db="bees.db")
    result = op.read_as_pandas(sql_query=sql_query)
    result = {
        "reply": reply,
        "result": result.to_dict("records"),
    }

    if verbose >= 3:
        logger.info("Final result:\n%s" % json.dumps(result, indent=4))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_model")
    parser.add_argument("--prompt")
    parser.add_argument("--verbose", type=int, default=1)
    args = vars(parser.parse_args())

    logger.info("Command line args:\n%s" % json.dumps(args, indent=4))
    main(**args)
    logger.info("ALL DONE!\n")
