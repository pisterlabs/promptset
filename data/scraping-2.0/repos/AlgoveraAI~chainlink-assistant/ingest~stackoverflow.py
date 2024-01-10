import os
import re
import time
import pickle
import requests
import html2text
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from langchain.docstore.document import Document

from config import DATA_DIR, get_logger

logger = get_logger(__name__)


# Settings for stackoverflow requests
key = os.getenv("SO_KEY")
tag = "chainlink"
pagesize = 99
from_date = "1609459200"

# Today's int date
to_date = int(datetime.now().timestamp())

# https://api.stackexchange.com/docs/questions
def get_questions(tag, page, pagesize, from_date, to_date, key, access_token):
    # url = f'https://api.stackexchange.com/2.3/questions?order=desc&sort=creation&tagged={tag}&site={site}&pagesize={pagesize}&key={key}&access_token={access_token}&filter=withbody'
    url = f"https://api.stackexchange.com/2.3/questions?page={page}&pagesize={pagesize}&tagged={tag}&fromdate={from_date}&todate={to_date}&key={key}&access_token={access_token}&order=desc&sort=activity&site=stackoverflow&filter=withbody"
    response = requests.get(url)
    return response.json()["items"]


def get_answers(question_id, key, access_token):
    url = f"https://api.stackexchange.com/2.3/questions/{question_id}/answers?order=desc&sort=activity&site=stackoverflow&key={key}&access_token={access_token}&filter=withbody"
    response = requests.get(url)
    return response.json()["items"]


def get_all_questions(access_token):
    all_questions = None
    for i in range(1, 5):
        page = i
        if all_questions is None:
            all_questions = get_questions(
                tag=tag,
                pagesize=pagesize,
                key=key,
                access_token=access_token,
                from_date=from_date,
                to_date=to_date,
                page=page,
            )
        else:
            all_questions.extend(
                get_questions(
                    tag=tag,
                    pagesize=pagesize,
                    key=key,
                    access_token=access_token,
                    from_date=from_date,
                    to_date=to_date,
                    page=page,
                )
            )
        time.sleep(3)

    all_questions = sorted(
        all_questions, key=lambda x: x["creation_date"], reverse=True
    )

    # Only consider questions from user with reputation > 50. Some questions don't have reputation field. They should be removed
    all_questions = [
        question
        for question in all_questions
        if "reputation" in question["owner"].keys()
        and question["owner"]["reputation"] > 50
    ]

    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.ignore_emphasis = False
    h.ignore_tables = False
    questions_and_answers = {}

    for question in tqdm(all_questions, total=len(all_questions)):
        answers = get_answers(question["question_id"], key, access_token)
        if not answers:  # Skip questions with no answers
            continue

        # Only keep accepted answer
        answers = [answer for answer in answers if answer["is_accepted"] == True]

        # Only keep answer from user with reputation > 50; some answers don't have reputation field. they should be removed
        answers = [
            answer
            for answer in answers
            if "reputation" in answer["owner"].keys()
            and answer["owner"]["reputation"] > 50
        ]

        question_body = re.sub(r"!\[.*?\]\(.*?\)", "", h.handle(question["body"]))
        question_str = (
            "Question: (Asked on: "
            + datetime.utcfromtimestamp(question["creation_date"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            + ")\n"
            + question["title"]
            + "\n"
        )
        question_url = "URL: " + question["link"] + "\n\n"
        question_body_str = "Question Body:\n" + question_body + "\n\n"
        answers_str = (
            "Answers:\n"
            + "\n---\n".join(
                [
                    "(Answered on: "
                    + datetime.utcfromtimestamp(answer["creation_date"]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    + ")\n"
                    + re.sub(r"!\[.*?\]\(.*?\)", "", h.handle(answer["body"]))
                    for answer in answers
                ]
            )
            + "\n\n"
        )
        text = question_str + question_url + question_body_str + answers_str

        questions_and_answers[question["question_id"]] = Document(
            page_content=text,
            metadata={"source": question["link"], "type": "stackoverflow"},
        )

    logger.info("Scrapped stackoverflow")

    return questions_and_answers


def scrap_stackoverflow(access_token):
    logger.info(f"Key: {key}")
    logger.info(f"{access_token}")

    so_docs = get_all_questions(access_token)

    # Make sure data directory exists
    Path("./data").mkdir(parents=True, exist_ok=True)

    # Convert to list
    so_docs = list(so_docs.values())

    # Log first document
    logger.info(f"First document SO: {so_docs[0]}")

    # Save to disk
    with open(f"{DATA_DIR}/stackoverflow_documents.pkl", "wb") as f:
        pickle.dump(so_docs, f)

    logger.info("Scrapped stackoverflow")

    return so_docs
