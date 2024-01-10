from typing import Dict, List

import json
import os
import random
import time
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
import openai 
import sqlite3
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
import tqdm

from topics import TOPICS

@retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
def ask_gpt(prompt: str) -> str:
    time.sleep(1) # rate limit myself
    prompts_list = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = openai.ChatCompletion.create(
        # model="gpt-4",
        model="gpt-3.5-turbo",
        messages=prompts_list,
        max_tokens=64,
        temperature=0.1,
        top_p=1,
        frequency_penalty=0.25,
        presence_penalty=0,
    )
    return response["choices"][0]["message"]["content"]

def get_topics(paper: Dict) -> List[str]:
    topics_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(TOPICS))
    prompt = """Given the following twenty topics:
    {topics_str}
    
Please identify the top 1-5 topics that the following paper fits under. Return your answer as a python-style list, with quotes around each topic. Don't include the numbers in your answer. Choose the topics that really fit the paper, even if it's only a single one. DO NOT output anything except the list of topics, or someone will die.

Title: {title}

Abstract: {abstract}

Topics:
""".format(
        topics_str=topics_str,
        title=paper["title"].replace("\n", " "),
        abstract=paper["abstract"][:250].replace("\n", " ").strip() + "..."
    )
    answer = ask_gpt(prompt)

    try:
        topics = eval(answer)
    except SyntaxError as e:
        print("got bad answer", answer)
        raise e
    return topics



data_folder = '/Users/johnmorris/arxiv-gpt/data/'

# maintain separate databases for papers and topics. makes it easier
# to regenerate one or the other.
data_conn = sqlite3.connect(os.path.join(data_folder, 'database.db'), check_same_thread=False)
data_cursor = data_conn.cursor()

conn = sqlite3.connect(os.path.join(data_folder, 'topics.db'), check_same_thread=False)
cursor = conn.cursor()

lock = threading.Lock()

# table 3: topics
cursor.execute('''CREATE TABLE IF NOT EXISTS topics
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   topic TEXT,
                   paper_id TEXT)''')

def process_document(document):
    doc_id, date, doc_data = document
    doc_data = json.loads(doc_data)

    try:
        topics = get_topics(doc_data)
    except Exception as e:
        print(f"Error - {e}")
        print("Error – skipping document.")
        return

    for topic in topics:
        lock.acquire(True)
        cursor.execute("INSERT OR IGNORE INTO topics (topic, paper_id) VALUES (?,?)",
            (topic, doc_data["id"]))
        lock.release()


def main():
    # pull all documents
    # Get 5 random documents
    data_cursor.execute("SELECT * FROM documents")
    documents = data_cursor.fetchall()
    
    # Print the random documents
    executor = ThreadPoolExecutor(max_workers=5)

    with tqdm.tqdm(total=len(documents)) as pbar:
        futures = [executor.submit(process_document, doc) for doc in documents]
        for future in as_completed(futures):
            future.result()
            pbar.update(1)

            
    conn.commit()
    conn.close()
    print("done :-)")


if __name__ == '__main__':
    main()