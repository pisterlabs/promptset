import openai
import json, pickle, os
from util import *

from dotenv import load_dotenv
load_dotenv()
openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")

import logging
import logging.config
from yaml import safe_load

with open("../conf/logging.yml") as f:
    cfg = safe_load(f)
logging.config.dictConfig(cfg)
logger = logging.getLogger("main")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 

data_dir = "../../../data/clean"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def answer(info, question, category, entity_name, wikipedia_article, wikidata_text_by_relation):
    messages = []
    if info["name"]:
        messages.append({"role": "assistant", "content": f"Name of the {category}: {entity_name}"})
    if info["article"]:
        messages.append({"role": "assistant", "content": f"Article:\n{wikipedia_article}"})
    if info["relations"]:
        messages.append({"role": "assistant", "content": f"Relations:\n{wikidata_text_by_relation}"})
    messages.append({"role": "system", "content": f"You are a helpful QA bot."})
    messages.append({"role": "user", "content": f"Please answer the following question.\nQuestion: {question}"})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    return response.choices[0]["message"]["content"]

def answer_by_categories(categories):
    for category in categories:
        logger.info(f"category: {category}")
        category_dir = f"{data_dir}/{category}"
        with open(f"{category_dir}/qas.json") as f:
            qas = json.load(f) 
        with open(f"{category_dir}/id_to_name.pkl", 'rb') as f:
            id_to_name = pickle.load(f)
        with open(f"{category_dir}/wikidata.json") as f:
            wikidata = json.load(f)

        # for entity_id in ["Q47309330"]:
        for i, entity_id in enumerate(qas):
                logger.info(f"Answering questions for {entity_id} ({i+1}/{len(qas)})")
                with open(f"{category_dir}/wikipedia/{entity_id}.txt") as f:
                    wikipedia_article = f.read()
                for qa in qas[entity_id]:
                    patterns = [
                        {"name": False, "article": False, "relations": False, "confidence": False}, 
                        {"name": True, "article": False, "relations": False, "confidence": False}, 
                        {"name": True, "article": True, "relations": False, "confidence": False}, 
                        {"name": True, "article": False, "relations": True, "confidence": False}, 
                        {"name": True, "article": True, "relations": True, "confidence": False}, 
                        # {"name": True, "article": True, "relations": True, "confidence": True}, 
                    ]
                    for info in patterns:
                        try:
                            qa["A" + get_label(info)] = answer(info, qa["Q_rephrase_mask"], category, id_to_name[entity_id], customize_text(wikipedia_article), wikidata[entity_id]["text"])
                        except Exception as e:
                            logger.error(f"{entity_id}: {entity_id}")
                            logger.error(f"Error: {e}")

        with open(f"{category_dir}/qas.json", 'w') as f:
            json.dump(qas, f, indent=2)

def main():
    categories = ["aircraft"]
    # categories = ["aircraft", "athlete", "bird", "bread", "car", "director", "dog", "us_politician"]
    answer_by_categories(categories)

if __name__ == "__main__":
    main()