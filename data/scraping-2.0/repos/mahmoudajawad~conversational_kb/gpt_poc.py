import glob
import json
import logging
import math
import os
import pathlib
import sys

import openai
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity

from _shared import SYSTEM_PROMPT, num_tokens_from_messages

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

ARTICLES: dict[str, str] = {}
EMBEDDINGS: dict[str, list[float]] = {}
MESSAGES: list[dict[str, str]] = []
ARTICLES_MATCH: list[str] = []

MAX_KNOWLEDGE = 5


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    generate_articles_embeddings()

    logging.info("Articles and embeddings are loaded. AMA!")
    ama_loop()


def ama_loop() -> None:
    while True:
        logging.info("Empty or exit to exit")
        question = input("AMA: ").strip()
        if not question or question == "exit":
            logging.info("Talk to you later!")
            sys.exit(0)
        question_embeddings = get_embedding(question)
        embeddings_scores: dict[str, float] = {}
        logging.debug("Comparing question embeddings to articles..")
        for article, article_embeddings in EMBEDDINGS.items():
            embeddings_scores[article] = cosine_similarity(
                question_embeddings, article_embeddings
            )
        embeddings_scores_sorted = sorted(
            list(embeddings_scores.items()), key=lambda score: score[1], reverse=True
        )
        match_article = embeddings_scores_sorted[0][0]
        ARTICLES_MATCH.append(match_article)

        logging.debug("Embeddings scores: %s", embeddings_scores_sorted)
        logging.debug("Resorting to find answer from: %s", match_article)

        # Build knowledge:
        # Use top most matched article
        knowledge_count = 1
        knowledge = ARTICLES[match_article]
        # Also add last matched article to give GPT the chance to understand follow-up questions
        last_match_article = ARTICLES_MATCH[-1]
        if last_match_article != match_article:
            knowledge += f"\n{last_match_article}"
        # Loop over all articles scores to add up to MAX_KNOWLEDGE articles with match score > 0.85
        for article, score in embeddings_scores_sorted[1:]:
            if score >= 0.80:
                knowledge += f"\n{ARTICLES[article]}"
                knowledge_count += 1
            if knowledge_count > MAX_KNOWLEDGE:
                break

        logging.debug("Compiled knowledge of %s articles", knowledge_count)

        MESSAGES.append({"role": "user", "content": question})

        system_prompt = f"{SYSTEM_PROMPT}{knowledge}"
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]

        # Truncate system message to reach optimal 2500 tokens
        # TODO: Change into recursion
        knowledge_tokens_count = num_tokens_from_messages(messages)
        while True:
            if knowledge_tokens_count <= 2500:
                break
            # While tokens count, usually, more than words, truncate only number of excess words
            knowledge_words = messages[0]["content"].split(" ")
            knowledge_words_count = len(knowledge_words)
            messages[0]["content"] = " ".join(
                knowledge_words[
                    : math.floor(
                        (knowledge_words_count * 2500) / knowledge_tokens_count
                    )
                ]
            )
            knowledge_tokens_count = num_tokens_from_messages(messages)

        # Insert user question
        messages.append(MESSAGES[-1])

        # Loop in reverse over all messages (except last which is already added) and add as many as possible
        # without breaking token limit of 3500
        for i, message in enumerate(MESSAGES[1:-10:-1]):
            if num_tokens_from_messages([*messages, message]) < 3500:
                messages.insert(len(messages) - (i + 1), message)

        logging.debug("Messages to be sent to OpenAI: %s", messages)
        logging.debug("Total messages count is: %s", len(messages))

        answer = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
        )
        response_text = answer["choices"][0]["message"]["content"]
        MESSAGES.append({"role": "assistant", "content": response_text})
        print(f"> {response_text}")


def get_embedding(text: str) -> list[float]:
    result = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return result["data"][0]["embedding"]


def generate_articles_embeddings() -> None:
    logging.info("Attempting to check articles...")
    dirname = os.path.dirname(__file__)
    articles = glob.glob(os.path.join(dirname, "articles", "*.txt"))

    for article in articles:
        article_name = pathlib.Path(article).stem
        if os.path.isfile(f"embeddings/{article_name}.json"):
            logging.info(
                "Article '%s' embeddings already generated. Skipping..", article_name
            )

            with open(article, encoding="UTF-8") as f:
                ARTICLES[article_name] = f.read()

            with open(f"embeddings/{article_name}.json", encoding="UTF-8") as f:
                EMBEDDINGS[article_name] = json.loads(f.read())

            continue

        logging.info(
            "Article '%s' embeddings not generated. Generating..", article_name
        )
        with open(article, encoding="UTF-8") as f:
            article_content = f.read()
            ARTICLES[article_name] = article_content
            article_embeddings = get_embedding(article_content)
            EMBEDDINGS[article_name] = article_embeddings

        with open(f"embeddings/{article_name}.json", "w", encoding="UTF-8") as f:
            f.write(json.dumps(article_embeddings))

        logging.info("  - Done")
