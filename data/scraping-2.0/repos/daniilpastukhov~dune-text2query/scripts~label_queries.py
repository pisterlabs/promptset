import os
import openai
from scripts.templates import LABEL_QUERY_TEMPLATE
from scraper.dune_scraper.db import MongoDatabase
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def label_query(query: str) -> None:
    """Label a query with the help of GPT3 API.
    Args:
        query (str): A query.
    Returns:
        str: The query type.
    """
    gpt_prompt = LABEL_QUERY_TEMPLATE.format(query)
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=gpt_prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    db.update_label(query['_id'], response['choices'][0]['text'].strip())


if __name__ == '__main__':
    db = MongoDatabase()
    queries = db.get_queries()

    for q in tqdm(queries):
        if len(q['_id'].split()) < 200 and len(q['_id']) < 5000 and 'query_label' not in db.find_one(q['_id']):
            label_query(q)
