from dotenv import load_dotenv
import os
from openai import OpenAI
from pymongo import MongoClient
import math
from collections import defaultdict
import json
import time

TWEETS_DATABASE = "Tweets"
QUERY_DATABASE = "Queries"

load_dotenv()
openai_client = OpenAI()


def get_db_handle():
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_STRING"))
    db_handle = client[os.getenv("DB_NAME")]
    return db_handle, client


def create_json_matrix_prompt(statements_1, statements_2):
    intro_prompt = (
        "As an AI, you are tasked with evaluating the level of agreement or disagreement between two sets of statements. "
        "Your analysis should be rooted in a detailed and thoughtful examination of each statement, considering not only the direct content but also the underlying implications and contexts. "
        "For each statement pair, assign a score from -10 (indicating complete disagreement) to 10 (indicating complete agreement). "
        "This scoring should reflect a comprehensive understanding of how the statements relate, taking into account their broader meanings and potential connections or contradictions.\n\n"
        "Focus exclusively on the content and deeper meanings of the statements, avoiding any influence from ideological or philosophical biases. "
        "When statements do not explicitly agree or contradict but have deeper connections or oppositions, these should be carefully considered in your scoring.\n\n"
        "Examples:\n"
        "'Smartphones are essential for modern communication.' and 'Most people rely heavily on technology for daily tasks.' might score high, reflecting a thematic agreement in technology reliance.\n"
        "'Maintaining natural ecosystems is vital for biodiversity.' and 'Economic development should be prioritized over environmental concerns.' would likely score negatively, due to underlying opposition in priorities.\n\n"
        "Please present the scores in a JSON formatted matrix, using indices for the statements from each group. Here is the format for a matrix where each group has two statements:\n"
        "All responses should be formated in this sample json format:\n"
        '{}"matrix": [[0, 0], [0, 0]]{}\n\n'
        "This response will be used by a script, so it is of great importance that your response is nothing but just the json response, ***any text not in the json block will cause the script to fail***. \n"
        "do your thought process before you generate the matrix as comments and only as comments in the json block, and please be as concise as possible to minimize tokens utilization. and cost of execution"
        "Now, apply this approach to the following statements:\n"
        "Group 1 has {} statements and Group 2 has {} statements.\n"
        "Analyze the following statements:\n\nGroup 1:\n".format(
            "{", "}", len(statements_1), len(statements_2)
        )
    )

    for i, statement1 in enumerate(statements_1, start=1):
        intro_prompt += f"{i}. {statement1}\n"

    intro_prompt += "\nGroup 2:\n"

    for j, statement2 in enumerate(statements_2, start=1):
        intro_prompt += f"{j}. {statement2}\n"

    return intro_prompt


def get_similarity_score(statements_1, statements_2):
    chat_completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": create_json_matrix_prompt(statements_1, statements_2),
            }
        ],
        model="gpt-3.5-turbo",
        timeout=60,
    )
    return chat_completion.choices[0].message.content


def get_earliest_pending_query():
    db_handle, client = get_db_handle()
    queries_db = db_handle[QUERY_DATABASE]
    earliest_pending_query = queries_db.find_one(
        {"status": "pending"}, sort=[("timestamp", 1)]
    )
    client.close()
    return earliest_pending_query


def create_statement_2_list(query):
    statement_2_list = []
    query_dict = {}
    for category in query["query"]:
        for subcategory in query["query"][category]:
            statement_2_list.append(query["query"][category][subcategory])
            query_dict[len(statement_2_list) - 1] = (category, subcategory)
    # print(query_dict)
    return statement_2_list, query_dict


def get_num_pages(page_size):
    db_handle, client = get_db_handle()
    tweets = db_handle[TWEETS_DATABASE]
    num_tweets = tweets.count_documents({})
    client.close()
    return math.ceil(num_tweets / page_size)


def get_tweets(page_size, page_num):
    db_handle, client = get_db_handle()
    tweets = db_handle[TWEETS_DATABASE]
    tweets_cursor = tweets.find({}).skip(page_size * (page_num - 1)).limit(page_size)
    client.close()
    return tweets_cursor


def create_statement_1_list(tweets_cursor):
    statement_1_list = []
    author_dict = {}
    for user in tweets_cursor:
        for tweet in user["tweets"]:
            statement_1_list.append(tweet["content"])
            author_dict[len(statement_1_list) - 1] = user["uname"]
    return statement_1_list, author_dict


def compute_author_scores_by_statement_2(page_size, query):
    author_scores_of_statement_2 = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    num_pages = get_num_pages(page_size)
    for page_num in range(1, num_pages + 1):
        tweets_cursor = get_tweets(page_size, page_num)
        statement_1_list, author_dict = create_statement_1_list(tweets_cursor)
        statement_2_list, query_dict = create_statement_2_list(query)
        cnt = 0
        while True:
            try:
                cnt += 1
                similarity_score = get_similarity_score(
                    statement_1_list, statement_2_list
                )
                break
            except Exception as e:
                if cnt > 3:
                    print("[ERROR]: Exceeded 3 retries")
                    return None
                print("Failed, retrying in 30s...")
                print("[Exception]:", e)
                time.sleep(30)
                print("retrying...")
                continue
        if similarity_score.startswith("```json"):
            similarity_score = similarity_score[7:-3]
        # print(similarity_score)
        similarity_score = json.loads(similarity_score)
        for statement_1_index, statement_2_scores in enumerate(
            similarity_score["matrix"]
        ):
            for statement_2_index, score in enumerate(statement_2_scores):
                author_scores_of_statement_2[author_dict[statement_1_index]][
                    query_dict[statement_2_index][0]
                ][query_dict[statement_2_index][1]].append(score)
    return author_scores_of_statement_2


def average_author_scores_by_statement_2(author_scores_of_statement_2):
    author_scores_by_statement_2 = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for author in author_scores_of_statement_2:
        for category in author_scores_of_statement_2[author]:
            for subcategory in author_scores_of_statement_2[author][category]:
                author_scores_by_statement_2[author][category][subcategory] = sum(
                    author_scores_of_statement_2[author][category][subcategory]
                ) / len(author_scores_of_statement_2[author][category][subcategory])
    return author_scores_by_statement_2


def cluster_and_count(
    unames, categories, category_index, average_author_scores_by_statement_2
):
    if category_index == len(categories):
        return {"result": {}, "next_category_result": {}}
    result = defaultdict(lambda: 0)
    next_category_result = {}
    subcats = defaultdict(list)
    for uname in unames:
        for subcategory in average_author_scores_by_statement_2[uname][
            categories[category_index]
        ]:
            if (
                average_author_scores_by_statement_2[uname][categories[category_index]][
                    subcategory
                ]
                >= 0
            ):
                result[subcategory] += 1
                subcats[subcategory].append(uname)
    for subcategory in subcats:
        next_category_result[subcategory] = cluster_and_count(
            subcats[subcategory],
            categories,
            category_index + 1,
            average_author_scores_by_statement_2,
        )
    return {"result": result, "next_category_result": next_category_result}


def update_query_status(query, query_result):
    db_handle, client = get_db_handle()
    queries_db = db_handle[QUERY_DATABASE]
    queries_db.update_one(
        {"_id": query["_id"]}, {"$set": {"status": "processed", "result": query_result}}
    )
    client.close()


def process_query(query):
    author_scores_of_statement_2 = compute_author_scores_by_statement_2(20, query)
    average_author_scores_by_statement_2_res = average_author_scores_by_statement_2(
        author_scores_of_statement_2
    )
    query_result = cluster_and_count(
        average_author_scores_by_statement_2_res.keys(),
        query["categories"],
        0,
        average_author_scores_by_statement_2_res,
    )
    update_query_status(query, query_result)


def execute_queries():
    while True:
        try:
            query = get_earliest_pending_query()
            if query:
                print("[INFO]: Processing query - ", query["_id"])
                process_query(query)
                print("[SUCCESS]: Processed query - ", query["_id"])
            else:
                print("No pending queries")
        except Exception as e:
            print("[ERROR]:", e)
        time.sleep(1)


if __name__ == "__main__":
    execute_queries()
