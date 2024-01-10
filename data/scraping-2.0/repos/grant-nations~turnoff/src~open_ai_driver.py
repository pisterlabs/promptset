"""
This file contains the functions that interact with the OpenAI API to score articles, summarize them, and write a tweet.

NOTE! The functions in this file cost MONEY to RUN.
"""

import os
import openai
from dotenv import load_dotenv
import src.utils as utils
import json
import time

GPT_MODEL = "gpt-3.5-turbo-16k"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def summarize_top_articles():
    """
    Summarize the top articles from each source, saving the summaries to a file.
    """

    words_per_summary = 200
    num_articles_per_source = 3

    openai.api_key = OPENAI_API_KEY

    # | Get today's articles from files |
    # V                                 V

    today = utils.get_current_date_string()
    articles_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", today, "articles.json")

    top_articles_per_source_by_title = {}

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", today, "article_scores.json"), "r") as f:
        source_article_scores = json.load(f)

        for source, articles in source_article_scores.items():
            sorted_articles = sorted(
                articles, key=lambda article: utils.get_score_from_text(article["score"]),
                reverse=True)

            top_articles_per_source_by_title[source] = [a["title"] for a in sorted_articles][:num_articles_per_source]

    print(top_articles_per_source_by_title)

    articles_data = None

    with open(articles_path, "r") as f:
        articles_data = json.load(f)

    # | Get the summary of each article |
    # V                                 V

    article_summaries = {}

    for source_articles in articles_data:

        source = source_articles["source"]
        article_summaries[source] = []

        articles = source_articles["articles"]

        for article in articles:
            title = article["title"]

            if title not in top_articles_per_source_by_title[source]:
                continue

            user_str = f"Please provide a concise summary of the following news article while identifying any potential biases. Summarize all essential details within {words_per_summary} words."

            user_str += f"Title: {article['title']}\n"
            user_str += f"Text: {article['text']}\n\n"

            # the messages are the context for the AI; the first message is the AI's role
            messages = [
                #     {"role": "system",
                #         "content": ai_role},
                {"role": "user",
                    "content": user_str}
            ]

            print("Sending request to OpenAI:")
            print("User:")
            print(user_str)
            print("--------------------")

            # send the request to OpenAI
            response = openai.ChatCompletion.create(
                model=GPT_MODEL,
                messages=messages
            )

            response_text = response["choices"][0]["message"]["content"]

            # response_text = "fake response"

            print("Response:")
            print(response_text)
            print("--------------------")

            article_summaries[source].append({
                "title": title,
                "summary": response_text
            })

    # | Save the article descriptions to a file |
    # V                                         V

    article_summaries_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", today, "article_summaries.json")

    with open(article_summaries_path, "w") as f:
        json.dump(article_summaries, f, indent=2)


def get_article_scores():
    """
    Get the score for each article from OpenAI, saving the scores to a file.
    """

    openai.api_key = OPENAI_API_KEY

    # | Get today's articles from files |
    # V                                 V

    today = utils.get_current_date_string()
    articles_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", today, "articles.json")

    articles_data = None

    with open(articles_path, "r") as f:
        articles_data = json.load(f)

    # | Get the score for each article |
    # V                                V

    article_scores = {}

    try:
        for source_articles in articles_data:
            source = source_articles["source"]
            article_scores[source] = []

            articles = source_articles["articles"]

            for article in articles:
                time.sleep(1)  # limit the number of requests to OpenAI to 1 per second

                ai_role = "You are a language model assigned to score news articles based on their relevance to current US news events. Consider the importance of the topic, its impact on US society, the prominence of the events, and the level of public interest. Give higher scores to articles covering national politics, significant policy changes, major social issues, and international events affecting the US. For instance, an article about a new federal policy addressing climate change should receive a high score, while an article about a local bake sale should receive a lower score. Please provide scores that reflect the articles' actual importance to US news."

                user_str = f"Here's an article from {source}:\n"
                user_str += f"Title: {article['title']}\n"
                user_str += f"Text: {article['text']}\n\n"

                user_str += "Give this article a score from 0 to 100, returning only the score and no other text."

                # the messages are the context for the AI. the first message is the AI's role
                messages = [
                    {"role": "system",
                        "content": ai_role},
                    {"role": "user",
                        "content": user_str}
                ]

                print("Sending request to OpenAI:")
                print("User:")
                print(user_str)
                print("--------------------")

                # send the request to OpenAI
                response = openai.ChatCompletion.create(
                    model=GPT_MODEL,
                    messages=messages,
                )

                response_text = response["choices"][0]["message"]["content"]

                print("Response:")
                print(response_text)
                print("--------------------")

                article_scores[source].append({
                    "title": article["title"],
                    "score": response_text,
                })

    except Exception as e:
        print(e)

    finally:
        # | Save the article descriptions to a file |
        # V                                         V

        article_descriptions_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", today, "article_scores.json")

        with open(article_descriptions_path, "w") as f:
            json.dump(article_scores, f, indent=2)


def write_tweet_from_summaries():
    """
    Write a tweet from the summaries of the top articles from each source, saving the tweet to a file.
    """

    temp = 1.3

    openai.api_key = OPENAI_API_KEY

    today = utils.get_current_date_string()
    summaries_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", today, "article_summaries.json")

    summaries = None

    with open(summaries_path, "r") as f:
        summaries = json.load(f)

    ai_role = "You are an educated AI that views the world through the lense of existentialism. "
    # ai_role += "Generally, you find the divisiveness of humanity baffling, humorous, and often times sad. "
    ai_role += "You are employed as a reporter for an AI news agency that reports daily on human affairs."

    user_str = "Write a tweet about something you find noteworthy in today's human events. Below are summaries of some news articles from today; feel free to use them if necessary. Make sure to keep the tweet to within 280 characters.\n\n"

    for _, articles in summaries.items():
        for article in articles:
            user_str += f"Title: {article['title']}\n"
            user_str += f"Summary: {article['summary']}\n\n"

    messages = [
        {"role": "system",
            "content": ai_role},
        {"role": "user",
            "content": user_str}
    ]

    print("System:")
    print(ai_role)
    print("--------------------")
    print("Sending request to OpenAI:")
    print("User:")
    print(user_str)
    print("--------------------")

    # send the request to OpenAI
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=temp,
    )

    tweet = response["choices"][0]["message"]["content"]

    tweet = utils.remove_quotations(tweet)

    print("Response:")
    print(tweet)
    print("--------------------")

    print("Tweet length:", len(tweet))

    if len(tweet) > 280:
        # try to reduce the length of the tweet without prompting again
        tweet = utils.remove_hashtags(tweet)

    reduce_count = 0
    tweet_retry_attempts = 10

    while len(tweet) > 280:
        # continue the conversation with the AI to reduce the length of the tweet

        tweet = utils.remove_hashtags(tweet)

        if reduce_count > tweet_retry_attempts:
            print(f"Tweet still too long after {tweet_retry_attempts} attempts. Aborting...")
            break

        reduce_count += 1

        print("Tweet too long. Reducing length...")

        messages.append({"role": "assistant",
                         "content": tweet})
        messages.append({"role": "user",
                         "content": "That tweet was more than 280 characters. Please shorten it or write a new one that's within 280 characters."})

        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=temp,
        )

        tweet = response["choices"][0]["message"]["content"]

        tweet = utils.remove_quotations(tweet)

        print("Response:")
        print(tweet)

        print("Tweet length:", len(tweet))

    if len(tweet) <= 280:
        print("Tweet ok ", end="", flush=True)
        utils.print_green(utils.CHECK_MARK)
    else:
        print("Tweet too long ", end="", flush=True)
        utils.print_red("X")

    tweet_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", today, "tweet.txt")

    with open(tweet_path, "w") as f:
        f.write(tweet)


if __name__ == "__main__":
    get_article_scores()
