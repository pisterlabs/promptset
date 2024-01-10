from langchain.llms import OpenAI
import sqlite3
from typing import List


# get a token: https://platform.openai.com/account/api-keys
### MAKE SURE YOU DO NOT PUSH THIS TO THE GITHUB REPO
# your_open_ai_key = ""

# model_name = "gpt-3.5-turbo-instruct"

# llm = OpenAI(openai_api_key=your_open_ai_key, model_name=model_name, max_tokens=100)

con = sqlite3.connect("twitter.db")
cur = con.cursor()


def get_tweets(trend_name: str) -> List[List]:
    """
    function to get the tweets for a given trend

    trend_name: the name of the trend

    returns: a list of tweets
    """
    cur.execute(
        "SELECT * FROM tweets WHERE trend_id = (SELECT id FROM trends WHERE trending_topic_name = ?)",
        (trend_name,),
    )
    tweets = cur.fetchall()

    for i, tweet in enumerate(tweets):
        tweets[i] = list(tweet)

    return tweets


def calculate_tweet_score(tweet: List) -> int:
    """
    function to calculate the score of a tweet

    tweet: row from the tweets table

    returns int: the score of the tweet
    """

    # get the number of likes
    likes = tweet[3]

    # get the number of comments
    comments = tweet[4]

    # get the number of retweets
    retweets = tweet[5]

    # get the number of views
    views = tweet[6]

    # get the date tweeted
    date_tweeted = tweet[7]

    # get the date retrieved
    date_retrieved = tweet[8]

    # calculate the score
    # TODO: make this calculation better

    score = (
        string_to_int(likes) +
        string_to_int(comments) +
        string_to_int(retweets) +
        string_to_int(views)/1000
    )

    return score

def string_to_int(num):
    """
    function to convert a string to an int

    num: the string to convert

    returns: the int
    """
    if num.__contains__("."):
        return int(num.replace(".", "").replace(",", "").replace("K", "00").replace("M", "00000"))
    else:
        return int(num.replace(",", "").replace("K", "000").replace("M", "000000"))


def get_best_tweets(tweets: List = [], n: int = 3) -> List[List]:
    """
    function to impliment the sorting/scoring logic for the tweets that we give chatgpt
    rn just sorts by likes

    tweets: a list of tweets
    n: the number of tweets to return

    returns: a list of the top n tweets
    """

    if tweets == []:
        raise Exception("No tweets given")

    for i, tweet in enumerate(tweets):
        # calculate the score of the tweet
        tweets[i].append(calculate_tweet_score(tweet))

    # get the top 3 tweets based on the score
    tweets.sort(key=lambda x: x[-1], reverse=True)
    best_tweets = tweets[0:n]

    return best_tweets


def format_tweets(tweets: List[List]) -> List[str]:
    """
    format the each tweet with extra information as needed to give to chatgpt

    tweets: a list of tweets

    returns: a list of formatted tweets
    """
    template = "Tweet: {tweet} \nScore: {score}\n\n"

    for i, tweet in enumerate(tweets):
        tweets[i] = template.format(tweet=tweet[2], score=tweet[-1])

    return tweets


def generate_new_tweet(trend: str, best_tweets: List[List], llm) -> str:
    """
    function to generate a new tweet for a given trend

    trend: the trend to generate a new tweet for
    best_tweets: the top tweets for the trend

    returns: a new tweet
    """
    template = "You job is to generate a new tweet for the current trending topic based on the highest scoring tweets for that topic:{trend}\n\nHere are the top tweets for this topic and the scores for each tweet:\n\n{best_tweets}\n\nNow generate a new tweet for this topic based on these tweets but do not include a score:\n\n"

    # format the tweets
    best_tweets = format_tweets(best_tweets)

    best_tweets = "".join(best_tweets)

    # generate the prompt
    prompt = template.format(trend=trend, best_tweets=best_tweets)

    # generate a new tweet
    new_tweet = llm(prompt=prompt)

    return new_tweet


if __name__ == "__main__":
    # see all trends
    available_trends = cur.execute("SELECT * FROM trends")

    # print the trends
    print(available_trends.fetchall())

    # ask user to select a trend
    trend = input("Select a trend: ")

    tweets = get_tweets(trend)

    # get the best tweets
    best_tweets = get_best_tweets(tweets)

    # generate a new tweet
    new_tweet = generate_new_tweet(trend, best_tweets)

    print("The new tweet is:",new_tweet)
